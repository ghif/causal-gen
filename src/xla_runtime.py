import contextlib
import os
from typing import Any, Dict, Iterable

import torch


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_custom_scalars(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _load_xla():
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError(
            "TPU execution requires torch-xla. Install the matched torch/torch-xla "
            "versions from requirements-tpu.txt."
        ) from exc
    return xm


def _prepare_tpu_environment() -> None:
    """Set TPU env defaults required by torch-xla before importing it."""
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return

    # torch-xla consults the Cloud TPU metadata service unless this is set.
    # On TPU VMs the metadata lookup is sometimes unavailable from the runtime
    # environment, so we prefer env-based configuration when launching locally.
    os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

    # torch-xla's env-based path reads TPU_ACCELERATOR_TYPE, but the import-time
    # checks also expect ACCELERATOR_TYPE to exist. Mirror either one into the
    # other so both code paths work.
    accelerator_type = os.environ.get("ACCELERATOR_TYPE") or os.environ.get(
        "TPU_ACCELERATOR_TYPE"
    )
    if accelerator_type is None:
        # Conservative default for TPU VM v6e hosts. Users can override this by
        # exporting ACCELERATOR_TYPE or TPU_ACCELERATOR_TYPE before launching.
        accelerator_type = "v6e-8"

    os.environ.setdefault("ACCELERATOR_TYPE", accelerator_type)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", accelerator_type)

    # When metadata is unavailable, torch-xla also needs topology values that
    # are normally derived from the TPU VM environment. Use the number of local
    # TPU chips visible in sysfs as the single-host process bound.
    tpu_vendor = "0x1ae0"
    tpu_device_ids = {"0x0027", "0x005e", "0x0063", "0x006f", "0x0056", "0x0062"}
    num_chips = 0
    try:
        for device_name in os.listdir("/sys/bus/pci/devices"):
            device_dir = os.path.join("/sys/bus/pci/devices", device_name)
            vendor_file = os.path.join(device_dir, "vendor")
            device_file = os.path.join(device_dir, "device")
            if not os.path.isfile(vendor_file) or not os.path.isfile(device_file):
                continue
            with open(vendor_file) as f:
                vendor_id = f.read().strip()
            if vendor_id != tpu_vendor:
                continue
            with open(device_file) as f:
                device_id = f.read().strip()
            if device_id in tpu_device_ids:
                num_chips += 1
    except OSError:
        num_chips = 0

    if num_chips <= 0:
        num_chips = 1

    worker_host = "127.0.0.1"
    worker_hosts = ",".join([worker_host] * num_chips)
    os.environ.setdefault("TPU_WORKER_HOSTNAMES", worker_hosts)
    os.environ.setdefault("TPU_WORKER_ID", "0")

    # Some libtpu/PJRT builds expect an explicit worker-address list even on a
    # single-host slice. Synthesize the local host layout with one port per chip.
    process_ports = [8476 + i for i in range(num_chips)]
    process_addresses = ",".join(f"{worker_host}:{port}" for port in process_ports)
    os.environ.setdefault("TPU_PROCESS_ADDRESSES", process_addresses)

    if os.environ.get("TPU_PROCESS_BOUNDS") is None:
        os.environ["TPU_PROCESS_BOUNDS"] = f"{num_chips},1,1"

    os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1")


def tpu_environment_available() -> bool:
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        try:
            _load_xla()
            return True
        except RuntimeError:
            return False
    return False


def is_xla_device(device: torch.device) -> bool:
    return device.type == "xla"


def xla_device() -> torch.device:
    return _load_xla().xla_device()


def world_size() -> int:
    try:
        import torch_xla.runtime as xr

        return xr.world_size()
    except (ImportError, AttributeError):
        return _load_xla().xrt_world_size()


def rank() -> int:
    try:
        import torch_xla.runtime as xr

        return xr.global_ordinal()
    except (ImportError, AttributeError):
        return _load_xla().get_ordinal()


def is_master() -> bool:
    return _load_xla().is_master_ordinal(local=False)


def rendezvous(tag: str) -> None:
    _load_xla().rendezvous(tag)


def optimizer_step(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().optimizer_step(optimizer, barrier=False)
    else:
        optimizer.step()


def mark_step(device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().mark_step()


def synchronize(device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().mark_step()
        _load_xla().wait_device_ops()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronize_fn = getattr(torch.mps, "synchronize", None)
        if synchronize_fn is not None:
            synchronize_fn()


def reduce_sum(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if is_xla_device(device):
        return _load_xla().all_reduce(
            _load_xla().REDUCE_SUM, value, scale=1.0
        )
    return value


def reduce_stats(stats: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if not is_xla_device(device):
        return stats

    keys = list(stats)
    values = []
    for key in keys:
        value = stats[key]
        if torch.is_tensor(value):
            value = value.detach().to(device=device, dtype=torch.float32)
        else:
            value = torch.tensor(float(value), device=device, dtype=torch.float32)
        values.append(value)
    reduced = reduce_sum(torch.stack(values), device)
    return {key: reduced[index] for index, key in enumerate(keys)}


def autocast(device: torch.device, precision: str):
    use_bf16 = is_xla_device(device) and precision in {"auto", "bf16"}
    if use_bf16:
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def wrap_loader(loader: Iterable, device: torch.device) -> Iterable:
    if not is_xla_device(device):
        return loader
    from torch_xla.distributed.parallel_loader import MpDeviceLoader

    return MpDeviceLoader(loader, device)


def launch(function, args=(), debug_single_process: bool = False):
    _prepare_tpu_environment()
    try:
        import torch_xla
    except ImportError:
        _load_xla()
    return torch_xla.launch(
        function, args=args, debug_single_process=debug_single_process
    )


def save(data: Any, path: str, device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().save(data, path, master_only=False)
    else:
        torch.save(data, path)
