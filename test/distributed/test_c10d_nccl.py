# Owner(s): ["oncall: distributed"]

import copy
import json
import os
import pickle
import random
import re
import signal
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import auto, Enum
from itertools import chain, product
from unittest import mock, SkipTest

import torch
import torch.distributed as c10d
import torch.distributed._functional_collectives as _functional_collectives


if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


import test_c10d_common
from test_c10d_common import (
    ConvNet,
    DoubleGpuNet,
    FFTModel,
    gpus_for_rank,
    ModuleForDdpCommHook,
)

import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._C._distributed_c10d import ErrorType, OpType, WorkResult
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_cuda import _get_torch_rocm_version, TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    get_timeout,
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_multicast_support,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
    sm_is_or_higher_than,
    TEST_SKIPS,
    with_dist_debug_levels,
    with_nccl_blocking_wait,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_SANDCASTLE,
    MI300_ARCH,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    runOnRocmArch,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

# bfloat16 is only supported by CUDA 11+
BFLOAT16_AVAILABLE = torch.cuda.is_available() and (
    torch.version.cuda is not None or torch.version.hip is not None
)


class ProcessGroupNCCLNoGPUTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        pass

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(TEST_CUDA, "GPUs are available, skipping test")
    def test_init_no_gpus(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(
            ValueError, "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"
        ):
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)


class ProcessGroupNCCLInitTest(MultiProcessTestCase):
    device_type = "cuda"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        dm = torch.get_device_module(self.device_type)
        return dm.device_count()

    @property
    def device(self):
        return torch.device(self.device_type, self.rank % self.world_size)

    # A helper with the must-needed init args for test infra.
    # kwargs can be filled in by individual init tests.
    def _init_process_group(self, **kwargs):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            **kwargs,
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_init_wo_backend_str(self):
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_scalable_init(self):
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = "1"
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = "0"


class ProcessGroupNCCLGroupTest(MultiProcessTestCase):
    def _create_process_group_nccl(self, store, opts, device_id=None):
        # create nccl processgroup with opts
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()

        # These tests are expected to throw SIGABRT(6);
        # But if we are in Sandcastle, `skip_but_pass_in_sandcastle` would return 0.
        TEST_NAN_ASSERT_RETURN = 0 if IS_SANDCASTLE else signal.SIGABRT
        self.special_return_code_checks = {
            self.test_nan_assert_float16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float32.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float64.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_bfloat16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e4m3fn.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e5m2.__wrapped__: TEST_NAN_ASSERT_RETURN,
        }

        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "nccl")

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # This TestCase focuses on creation, destroy and abort of PG's. So it
        # does not need auto-destroy upon exit.
        return False

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 1 GPU")
    @skip_if_lt_x_gpu(1)
    def test_nccl_dist_backend_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())

        # Both rank 0 and 1 will use the same CUDA device resulting in ncclInvalidUsage
        with self.assertRaises(dist.DistBackendError) as cm:
            dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)
        self.assertTrue(isinstance(cm.exception, dist.DistError))

        self.assertIsInstance(cm.exception, RuntimeError)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)

        def abortpg():
            c10d.distributed_c10d._get_default_group()._get_backend(
                torch.device(device)
            ).abort()

        # Initialize DDP to ensure "destroy_process_group" will not call
        # ProcessGroupNCCL destructor since DDP holds a reference to process group.
        # Run a single iteration of DDP to initialize state.
        model = DistributedDataParallel(
            torch.nn.Linear(10, 10).to(device), device_ids=[device]
        )
        model(t).sum().backward()

        # Now simulate collective getting stuck and abort gets us unstuck
        if self.rank == 0:
            dist.all_reduce(t)

            # Schedule thread before we get stuck to abort pg.
            thread = threading.Thread(target=abortpg)
            thread.start()

            # We would get stuck here due to d2h if we didn't abort.
            t.cpu()

            thread.join()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_close_pg(self, eager_init: bool):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)

        # Destroy pg and validate pg is no longer valid
        dist.destroy_process_group()
        with self.assertRaises(ValueError):
            dist.all_reduce(t)

    @requires_nccl()
    @skip_if_rocm_multiprocess
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_restart_pg(self):
        # Note: restart test passes steadily only for blocking mode for now.
        # TODO: expand this test to non-blocking mode
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        # initialize pg for the first time
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        t0 = torch.rand(10, 10, device=device)
        # First allreduce to lazy initialize default pg
        dist.all_reduce(t0)
        torch.cuda.synchronize()
        # Destroy pg
        dist.destroy_process_group()

        # we need a new Store for the new PG, achieving it by adding prefix
        new_store = c10d.PrefixStore("2nd", store)

        # re-initialize pg
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=new_store,
        )
        t1 = torch.rand(5, 5, device=device)
        dist.all_reduce(t1)
        torch.cuda.synchronize()
        dist.destroy_process_group()
        # validate default pg is no longer valid
        with self.assertRaises(ValueError):
            dist.all_reduce(t1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_cuda_event_cache_mthd_race(self):
        # This unit test is to test the case when the collective is launched in
        # a side thread and the thread dies before the cache has been fully recycled.
        # More details can be found in this issue: https://github.com/pytorch/pytorch/issues/143470.

        # initiate collectives here
        def init_collective_task(t):
            dist.all_reduce(t)
            dist.all_reduce(t)
            dist.all_reduce(t)

        os.environ["TORCH_NCCL_CUDA_EVENT_CACHE"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)
        dist.all_reduce(t)
        dist.all_reduce(t)
        side_thread = threading.Thread(target=init_collective_task, args=(t,))
        side_thread.start()
        side_thread.join()
        torch.cuda.synchronize()

        # reset ENV
        os.environ["TORCH_NCCL_CUDA_EVENT_CACHE"] = "0"

    CUDA_12_AND_ABOVE = torch.cuda.is_available() and (
        torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 12
    )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        # skip for cu126 as well due to https://github.com/pytorch/pytorch/issues/153479
        not (TEST_MULTIGPU and CUDA_12_AND_ABOVE),
        "NCCL test requires 2+ GPUs and Device side assert could cause unexpected errors in lower versions of CUDA",
    )
    @parametrize(
        "type",
        [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    )
    @skip_if_rocm_multiprocess
    def test_nan_assert(self, type):
        # Expecting a device-side error when NaN is detected
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device("cuda"))

        device = self.rank_to_GPU[self.rank][0]
        # Cover different buffer sizes
        if type == torch.float64:
            size = (1024,)  # 1K elements
        elif type == torch.float32:
            size = (1024, 1024)  # 1M elements
        elif type == torch.float16:
            size = (1024, 1024, 1024)  # 1G elements
        else:
            size = (1,)  # 1 element

        # Note: currently we cannot fill values into a FP8 tensor, thus we
        # create the NaN tensor in float32 type and cast it to FP8
        if type == torch.float8_e4m3fn or type == torch.float8_e5m2:
            init_type = torch.float32
        else:
            init_type = type

        nan_tensor = torch.zeros(*size, dtype=init_type, device=device)
        # randomly pick an nan element
        index = tuple([random.randrange(size[i]) for i in range(len(size))])
        nan_tensor[index] = float("nan")
        if init_type != type:
            # Now cast to the targeted dtype
            nan_tensor = nan_tensor.to(type)

        output = torch.empty(self.world_size, *size, dtype=type, device=device)

        # confirm enable/disable flag works
        backend._set_enable_nan_check(False)
        # Note: using all-gather here bc some NCCL/SM version does not support
        # FP8 reduction
        # temporarily skip due to https://github.com/pytorch/pytorch/issues/153479
        # pg._allgather_base(output, nan_tensor)

        backend._set_enable_nan_check(True)
        try:
            pg._allgather_base(output, nan_tensor)
        except Exception:
            sys.exit(signal.SIGABRT)

        dist.destroy_process_group()

        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_rank_filter(self):
        # Putting NaN at recv buffer, program should not fail as NaN checker
        # should not check on receive buffer
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank:d}")
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        t = torch.ones(3, 4, dtype=torch.bfloat16, device=device)
        if self.rank != 0:
            # Putting NaN at recv buffer
            t[1, 1] = float("nan")
        # Against broadcast
        c10d.broadcast(t, 0)
        # Against P2P
        if self.rank == 0:
            c10d.send(t, 1)
        elif self.rank == 1:
            c10d.recv(t, 0)
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_check(self):
        # Not expecting an error, NaN check should not make legit code fail
        device = torch.device(f"cuda:{self.rank:d}")
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        x = torch.ones((10,), device=device) * self.rank
        t = torch.ones(3, 4, device=device)
        c10d.broadcast(x, src=0)
        c10d.all_reduce(t)
        c10d.barrier()
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    def _helper_test_extra_cuda_context_by_nvml(self):
        """
        A helper for `test_extra_cuda_context`, if pynvml is available.
        pynvml provides python bindings for NVIDIA NVML functionalities.
        Here we are interested in: nvmlDeviceGetComputeRunningProcesses
        """
        import pynvml

        pynvml.nvmlInit()

        device = torch.device(f"cuda:{self.rank:d}")
        x = torch.empty((1,), device=device)
        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
        del work
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        nprocs = len(processes)

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.cuda.synchronize(device)
        c10d.destroy_process_group()
        self.assertLessEqual(
            nprocs,
            1,
            f"Found {nprocs} processes creating contexts on {device}, expecting 1 at most",
        )

    def _helper_test_extra_cuda_context_by_memory(self):
        """
        A helper for `test_extra_cuda_context`, if pynvml is NOT available.
        If extra context is created, it would manifest into device 0's memory usage.
        """
        device = torch.device(f"cuda:{self.rank:d}")
        x = torch.empty((1,), device=device)
        # Rank 0 takes a snapshot before collective -- this snapshot should have
        # included rank 0's own context.
        if self.rank == 0:
            free, total = torch.cuda.mem_get_info(device)
            used_before = float(total - free)

        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
            free, total = torch.cuda.mem_get_info(device)
            used_after = float(total - free)
        del work

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.cuda.synchronize(device)
        c10d.destroy_process_group()
        if self.rank == 0:
            # If non-0 rank creates a context on device 0, this assert would
            # fail because one context takes about 1 GB -- much more than the
            # tensor size created in this test.
            self.assertTrue(
                # Bump the heuristic from 1.5 to 1.7 due to
                # https://github.com/pytorch/pytorch/issues/153122
                used_after < used_before * 1.7,
                f"{device} used {used_after} bytes after collective, "
                f"70% more than the status before ({used_before} bytes). "
                f"Extra CUDA context may have been created.",
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_cuda_context(self):
        # Check if non-0 ranks would create extra CUDA context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank:d}")
        c10d.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        try:
            self._helper_test_extra_cuda_context_by_nvml()
        except ModuleNotFoundError:
            self._helper_test_extra_cuda_context_by_memory()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_cuda_context_sync_ops(self):
        # Loop a bunch of sync ops and see if any of them creates extra context.
        # Requires nvml to check number of processes resident on a device.
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:
            self.skipTest("pynvml not available")

        # Check if non-0 ranks would create extra CUDA context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank:d}")
        c10d.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )

        x = torch.empty((1,), device=device)
        y = torch.empty((self.world_size,), device=device)

        c10d.all_reduce(x)
        c10d.reduce(x, dst=0)
        c10d.broadcast(x, src=0)
        c10d.all_gather_into_tensor(y, x)
        c10d.reduce_scatter_tensor(x, y)
        c10d.barrier()

        # Wait a bit for remote processes to touch my device
        if self.rank == 0:
            time.sleep(5)

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        nprocs = len(processes)

        # Don't exit till rank 0 is done with the nvml detection
        c10d.barrier()
        c10d.destroy_process_group()
        self.assertLessEqual(
            nprocs,
            1,
            f"Found {nprocs} processes creating contexts on {device}, expecting 1 at most",
        )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_destruct_before_terminate_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)
        # force destruction before terminating comms, destructor would terminate comms
        del pg

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_in_destroy_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)

        # Destroy pg and validate pg is NOT in working condition since
        # we have shutdown comms
        dist.destroy_process_group()
        with self.assertRaises(dist.DistBackendError):
            pg.allreduce([t])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_multi_pgs(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t1 = torch.rand(10, 10, device=device)
        t2 = torch.rand(10, 10, device=device)
        new_pg1.allreduce(t1).wait()
        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 0 because
        # it's not eager initialized
        self.assertEqual(backend.comm_split_count(), 0)
        # shutdown all NCCL PGs in one shot
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_mixed_empty_pgs(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        # PG1 is an PG without comms initialized, since we don't call collective on it
        new_pg1 = c10d.new_group([0, 1])  # noqa: F841
        new_pg2 = c10d.new_group([0, 1])
        t2 = torch.rand(10, 10, device=device)

        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 0
        self.assertEqual(backend.comm_split_count(), 0)
        # shutdown all NCCL PGs in one shot
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_file_store_check(self):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        # FileStore check() would be executed
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "0"

        # self.file_name is created using "delete=False"
        # e.g., self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(pg.rank(), self.rank)
        self.assertEqual(pg.size(), self.world_size)
        # give enough time for check() to be executed multiple times
        time.sleep(2)
        dist.destroy_process_group()

    def _check_nccl_timeout(self, expected_timeout):
        pg = dist.distributed_c10d._get_default_group()
        options = pg._get_backend(torch.device(f"cuda:{self.rank}")).options
        self.assertEqual(options._timeout, expected_timeout)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_init_process_group_nccl_timeout(self):
        # nccl is handled 'specially' inside init_process_group and its options class is different from the options
        # used by the other PG's.  There are specific edge cases for nccl that need to be tested.

        store = c10d.FileStore(self.file_name, self.world_size)
        base_opts = dict(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )

        # test the default value coming from the `init_process_group` kwarg default
        dist.init_process_group(**base_opts)
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # test that `kwarg` timeout takes effect
        new_timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, timeout=new_timeout)
        self._check_nccl_timeout(new_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg (or its kwdefault) taking precedence
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        with warnings.catch_warnings(record=True):
            dist.init_process_group(**base_opts, pg_options=opts)
            # TODO(whc) i verified that we are indeed emitting this warning, and i can't figure out why i can't catch it.
            # self.assertEqual(len(w), 1)
            # self.assertTrue("pg_options._timeout was specified" in str(w[-1].message))
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg taking precedence
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        dist.init_process_group(
            **base_opts, pg_options=opts, timeout=timedelta(seconds=1240)
        )
        self._check_nccl_timeout(timedelta(seconds=1240))
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    def test_set_nccl_pg_timeout(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        dist.init_process_group(**opts)
        pg = dist.distributed_c10d._get_default_group()
        pg.allreduce(torch.rand(10).cuda(self.rank))
        self._check_nccl_timeout(timedelta(seconds=123))
        pg._get_backend(torch.device(f"cuda:{self.rank}"))._set_default_timeout(
            timedelta(seconds=23)
        )
        self._check_nccl_timeout(timedelta(seconds=23))
        pg.allreduce(torch.rand(10).cuda(self.rank))
        c10d.distributed_c10d._set_pg_timeout(timedelta(seconds=252), pg)
        self._check_nccl_timeout(timedelta(seconds=252))

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    def test_extend_nccl_pg_timeout(self, backend):
        torch.cuda.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        dist.init_process_group(**opts)
        pg = dist.distributed_c10d._get_default_group()
        bankend = pg._get_backend(torch.device(f"cuda:{self.rank}"))
        w = pg.allreduce(torch.rand(10).cuda(self.rank))
        self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=123)))
        w.wait()
        bankend._set_default_timeout(timedelta(seconds=3))
        if self.rank == 0:
            # Ideally we want to sleep for a very long time, but this is not
            # feasible in unit test. So this is only a very tiny case.
            time.sleep(5)
            pg.allreduce(torch.rand(10).cuda(self.rank))
            time.sleep(5)
            pg.allreduce(torch.rand(5).cuda(self.rank))
            w = pg.allreduce(torch.rand(10).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=3)))
            w.wait()
        else:
            dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                timedelta(seconds=10)
            )
            w1 = pg.allreduce(torch.rand(10).cuda(self.rank))
            w2 = pg.allreduce(torch.rand(5).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w1, timedelta(seconds=13)))
            self.assertTrue(bankend._verify_work_timeout(w2, timedelta(seconds=13)))
            w1.wait()
            dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                timedelta(seconds=5)
            )
            # Since we are not block wait so use a sync here to leave enough time
            # for watchdog to reset first timeout extension.
            torch.cuda.synchronize(torch.device(f"cuda:{self.rank}"))
            w = pg.allreduce(torch.rand(10).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=8)))
            w.wait()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_new_group(self, eager_init: bool):
        # Test the optimization of new groups that contain all world
        # ranks use the "transparent" `ncclCommSplit` optimization.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )
        ng = c10d.new_group()
        tensor = torch.tensor([self.rank], device=device)
        dist.broadcast(tensor, 0)
        dist.broadcast(tensor, 0, group=ng)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.nccl.version()[-1] == "x", "NCCL test not for NCCLX"
    )
    def test_comm_split_subgroup(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        original_tensor = tensor.clone()
        ng = c10d.new_group([0])

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        if self.rank == 0:
            dist.broadcast(tensor, 0, group=ng)

        # no additional comm split happens after a collective.
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(tensor, original_tensor)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_eager_init_subgroup(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # default PG comm is not initialized yet
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend._is_initialized(), False)
        # create a subgroup eagerly
        new_group = c10d.new_group([0, 1], device_id=device)
        tensor = torch.full((1,), self.rank).cuda(device)
        dist.broadcast(tensor, 0, group=new_group)
        # the default group should stay lazy
        self.assertEqual(backend._is_initialized(), False)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_split_group(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        # Create subgroup between ranks 0, 1
        subg_ranks = [0, 1]
        ng1 = c10d.split_group(pg, [subg_ranks])
        backend1 = ng1._get_backend(torch.device(device))

        # check basic options are the same between parent and child
        self.assertEqual(backend.options._timeout, backend1.options._timeout)
        self.assertEqual(
            backend.options.is_high_priority_stream,
            backend1.options.is_high_priority_stream,
        )
        self.assertEqual(ng1.group_desc, "default_pg:split:0")

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        # dist.get_process_group_ranks returns the global ranks in the subgroup.
        self.assertEqual(
            dist.get_process_group_ranks(ng1),
            subg_ranks if self.rank in subg_ranks else [],
        )

        # is part of ng1; otherwise, -1
        if dist.get_rank(ng1) >= 0:
            dist.broadcast(tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(tensor, torch.full((1,), 0))

        ng2 = c10d.split_group(pg, [subg_ranks])
        self.assertEqual(ng2.group_desc, "default_pg:split:1")
        self.assertEqual(backend.comm_split_count(), 2)

        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_init(self):
        # Test creating a pg using nonblocking mode but not eagerly
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = self.rank_to_GPU[self.rank][0]
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 0)
        reduce_tensor = torch.rand(10, 10, device=device)
        # Run an allreduce, which should trigger a comm init for pg
        pg.allreduce(reduce_tensor).wait()
        new_pg = c10d.new_group()
        # even after pg's collective call, new pg's comm is not initialized until its own collectcive calls
        self.assertEqual(backend.comm_split_count(), 0)
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        self.assertEqual(backend.comm_split_count(), 0)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_with_eager_init(self):
        # Test creating a pg eagerly with nonblocking mode when
        # we've passed a specific device_id to init_process_group.
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # bound device to trigger eager init mode
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 0)
        reduce_tensor = torch.rand(10, 10, device=device)
        # Run an allreduce, comm should have already started initilizaing,
        # but allreduce is issued to CUDA STREAM only after the initialization is a success
        pg.allreduce(reduce_tensor).wait()
        new_pg = c10d.new_group()
        # new pg's comm is initialized eagerly
        self.assertEqual(backend.comm_split_count(), 1)
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        self.assertEqual(backend.comm_split_count(), 1)
        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_p2p(self):
        # Test creating a pg using nonblocking mode but not eagerly
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = self.rank_to_GPU[self.rank][0]
        self._create_process_group_nccl(store, self.opts())
        # Generate the same tensor
        send_tensor = torch.ones(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)
        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_subgroup_p2p(self, eager_init: bool):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )
        send_tensor = torch.ones(10, 10, device=device)
        group = dist.new_group()
        if self.rank == 0:
            dist.send(send_tensor, 1, group=group)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0, group=group)
            self.assertEqual(send_tensor, recv_tensor)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_get_uid(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        from torch.distributed.distributed_c10d import _get_process_group_uid

        self.assertEqual(_get_process_group_uid(pg), 0)
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(_get_process_group_uid(pg_2), 1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_set_process_group_desc(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg_default = self._create_process_group_nccl(
            store, self.opts(), device_id=device
        )
        self.assertEqual(pg_default.group_desc, "default_pg")
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        self.assertEqual(pg_1.group_desc, "test_purpose")
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(pg_2.group_desc, "undefined")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_deterministic_mode_no_break(self):
        torch.use_deterministic_algorithms(True)
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        self._create_process_group_nccl(store, self.opts(), device_id=device)
        tensor = torch.empty(10, 10, device=device)
        dist.all_reduce(tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_init_with_idx(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device_idx = self.rank
        dist.init_process_group(
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device_idx,
        )
        dist.all_reduce(torch.empty(1, device=torch.device("cuda", device_idx)))

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_block_current_stream(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)

        t = torch.rand(10, device=device)
        work = pg.allreduce(t)
        work.block_current_stream()

        torch.cuda.current_stream().synchronize()
        work.wait()
        torch.cuda.synchronize()


class NcclErrorHandlingTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 3

    @property
    def world_size(self):
        return 3

    @property
    def blocking_wait_error_msg(self):
        return "timeout"

    def _run_all_reduce(self, pg):
        pg.allreduce(torch.rand(10).cuda(self.rank))

    def _reduce_timeout(self):
        # set heartbeat timeout to a small value so that we don't wait too long
        # for things to shutdown
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "4"
        os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "1000"

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    @skip_but_pass_in_sandcastle("Test does not pass when run locally")
    def test_nccl_errors_nonblocking(self):
        self._reduce_timeout()
        # Note: we unset and restore TORCH_NCCL_ASYNC_ERROR_HANDLING for this test
        # since test_c10d_common runs with async error handling by default, but this
        # tests behavior when it is not enabled.
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            # This allreduce does not block Python thread as allreduce enqueues
            # the cuda operation, and then wait only blocks the current cuda
            # stream.
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()

            # Now the work scheduled next should hang forever since the previous
            # allreduce will never complete.
            t = threading.Thread(target=self._run_all_reduce, args=(process_group,))
            t.daemon = True
            t.start()
            t.join(int(get_timeout(self.id()) / 5))
            self.assertTrue(t.is_alive())

        if prev_nccl_async_error_handling is not None:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                prev_nccl_async_error_handling
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_blocking(self):
        self._reduce_timeout()
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
        )
        x = torch.rand(1024 * 1024).cuda(self.rank)
        process_group.allreduce(x)
        if self.rank == 0:
            work = process_group.allreduce(x)
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                work.wait(timeout=timedelta(seconds=self.op_timeout_sec))

    def _test_barrier_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        process_group.barrier().wait()
        if self.rank == 0:
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # It seems the error message would be different depending on
                # whether the test is run on CI machine and devGPU.  Skipping
                # the error message check to make both sides happy.
                process_group.barrier().wait(
                    timeout=timedelta(seconds=self.op_timeout_sec)
                )

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_blocking_wait_with_barrier(self):
        self._reduce_timeout()
        self._test_barrier_error()

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_non_blocking_wait_with_barrier(self):
        self._reduce_timeout()
        # test the barrier behavior in the non blocking wait setting
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        self._test_barrier_error()
        if prev_nccl_async_error_handling is not None:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                prev_nccl_async_error_handling
            )

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_error_detection_and_propagation(self):
        def assert_fut_success(fut):
            self.assertEqual(WorkResult(fut.value()), WorkResult.SUCCESS)

        # test the barrier behavior in the non blocking wait setting
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_PROPAGATE_ERROR"] = "1"
        # set heartbeat timeout to a small value so that we don't wait too long for things to shutdown
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "5"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=2),
        )
        self.assertEqual(process_group.get_error(), ErrorType.SUCCESS)
        barrier_work = process_group.barrier()
        barrier_work.wait()
        barrier_result = barrier_work.get_future_result().wait()
        self.assertEqual(WorkResult(barrier_result), WorkResult.SUCCESS)
        ar_work = process_group.allreduce(torch.rand(10).cuda(self.rank))
        ar_work.wait()
        fut = ar_work.get_future_result()
        # test adding a callback function
        fut.then(assert_fut_success)
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(WorkResult(result), WorkResult.TIMEOUT)
            self.assertEqual(process_group.get_error(), ErrorType.TIMEOUT)
        else:
            # other ranks not exiting before rank 0 timeout, this is to avoid
            # nccl error happening before rank 0 timeouts
            time.sleep(4)
            self.assertEqual(process_group.get_error(), ErrorType.REMOTE_ERROR)

        # Mimicking all ranks sensing the timeout, abort
        process_group.abort()

        if prev_nccl_async_error_handling is not None:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                prev_nccl_async_error_handling
            )

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(3)
    def test_restart_pg_after_error(self):
        self._reduce_timeout()
        # test the barrier behavior in the non blocking wait setting
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid FR dumping logic during restart
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
        # avoid watchdog thread interference
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_PROPAGATE_ERROR"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        # initialize pg for the first time
        c10d.init_process_group(
            "nccl",
            timeout=timedelta(seconds=2),
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        pg = c10d.distributed_c10d._get_default_group()
        nccl_backend = pg._get_backend(torch.device(device))
        self.assertEqual(nccl_backend.get_error(), ErrorType.SUCCESS)
        barrier_work = nccl_backend.barrier()
        barrier_work.wait()
        barrier_result = barrier_work.get_future_result().wait()
        self.assertEqual(WorkResult(barrier_result), WorkResult.SUCCESS)
        self.assertEqual(nccl_backend.get_error(), ErrorType.SUCCESS)
        if self.rank == 0:
            work = nccl_backend.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(WorkResult(result), WorkResult.TIMEOUT)
            self.assertEqual(nccl_backend.get_error(), ErrorType.TIMEOUT)
            # we need a brand new fileStore for the new PG
            # the new file name is shared through the old fileStore
            new_file_name = tempfile.NamedTemporaryFile(delete=False).name
            store.set("file", new_file_name)
        else:
            # other ranks not exiting before rank 0 timeout, this is to avoid
            # nccl error happening before rank 0 timeouts
            time.sleep(4)
            self.assertEqual(nccl_backend.get_error(), ErrorType.REMOTE_ERROR)
            new_file_name = store.get("file").decode()

        # all ranks restart using a new store after detecting the timeout error
        nccl_backend.abort()
        dist.destroy_process_group()

        new_store = c10d.FileStore(new_file_name, self.world_size)
        # re-initialize pg
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=new_store,
        )

        new_pg = c10d.distributed_c10d._get_default_group()
        new_nccl_backend = new_pg._get_backend(torch.device(device))
        t = torch.rand(5, 5, device=device)
        dist.all_reduce(t)
        self.assertEqual(new_nccl_backend.get_error(), ErrorType.SUCCESS)
        torch.cuda.synchronize()
        dist.destroy_process_group()

        # give some time for other ranks to exit first before destroying FileStore
        if self.rank == 0:
            time.sleep(4)
            os.remove(new_file_name)

        if prev_nccl_async_error_handling is not None:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                prev_nccl_async_error_handling
            )

    def _run_invalid_nccl_blocking_wait_env(self, val):
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = val
        store = c10d.FileStore(self.file_name, self.world_size)
        with self.assertRaises(RuntimeError):
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_invalid_nccl_blocking_wait_env(self):
        self._run_invalid_nccl_blocking_wait_env("abc")
        self._run_invalid_nccl_blocking_wait_env("-1")
        self._run_invalid_nccl_blocking_wait_env("2147483647")
        self._run_invalid_nccl_blocking_wait_env("4294967295")


class NcclUserBufferRegistrationTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        nccl_debug_file = tempfile.NamedTemporaryFile()
        os.environ["NCCL_ALGO"] = "NVLS"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "NVLS"
        if torch.cuda.nccl.version() >= (2, 24, 3):
            os.environ["NCCL_DEBUG_SUBSYS"] = "REG,TUNING"
        os.environ["NCCL_DEBUG_FILE"] = nccl_debug_file.name
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_nccl()
    @requires_nccl_version((2, 19), "Need NCCL 2.19 for user buffer registration")
    @skip_if_lt_x_gpu(4)
    @requires_multicast_support()
    def test_nccl_user_buffer_registration(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            device_id=device,
        )
        torch.cuda.set_device(self.rank)
        pg = c10d.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device(device))

        # Use NCCL memory allocator
        pool = torch.cuda.MemPool(backend.mem_allocator)

        # allocate memory with ncclMemAlloc
        with torch.cuda.use_mem_pool(pool):
            tensor = torch.arange(1024 * 1024 * 2, device=device)

        # register buffers to NCCL
        backend.register_mem_pool(pool)

        # allreduce now should use NVIDIA Switches
        pg.allreduce(tensor).wait()
        torch.cuda.synchronize(device=device)

        # de-register buffers from NCCL
        backend.deregister_mem_pool(pool)

        # clean up memory
        del tensor, pool

        with open(os.environ["NCCL_DEBUG_FILE"]) as f:
            nccl_debug_file_content = f.read()
            # if buffers were registered and NVLS reduction ran, NCCL_DEBUG
            # should show successful registration in debug output
            if torch.cuda.nccl.version() >= (2, 24, 3):
                self.assertRegex(
                    nccl_debug_file_content, "successfully registered NVLS"
                )
            else:
                self.assertRegex(nccl_debug_file_content, "local-registered")

    @requires_nccl()
    @requires_nccl_version((2, 27), "Need NCCL 2.27 for window registration")
    @skip_if_lt_x_gpu(4)
    @requires_multicast_support()
    def test_nccl_window_registration(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        with torch.cuda.device(device):
            # Eager init the nccl comm so that we don't implicitly create one during register_mem_pool
            c10d.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
                store=store,
                device_id=device,
            )
            pg = c10d.distributed_c10d._get_default_group()
            backend = pg._get_backend(torch.device(device))

            # Use NCCL memory allocator
            # enable symmetric memory usage in NCCL
            pool = torch.cuda.MemPool(backend.mem_allocator)

            # allocate memory with ncclMemAlloc
            # note: symmetric kernels are not available for dtypes like torch.int64
            with torch.cuda.use_mem_pool(pool):
                tensor = torch.arange(
                    1024 * 1024 * 2, device=device, dtype=torch.float32
                )

            # register buffers to NCCL
            backend.register_mem_pool(pool, symm=True)

            # allreduce now should use NVIDIA Switches
            pg.allreduce(tensor).wait()
            # check that further allocations are also registered
            with torch.cuda.use_mem_pool(pool):
                tensor = torch.arange(
                    1024 * 1024 * 2, device=device, dtype=torch.float32
                )
            pg.allreduce(tensor).wait()
            torch.cuda.synchronize(device=device)

            # de-register buffers from NCCL
            backend.deregister_mem_pool(pool)

            # clean up memory
            del tensor, pool

        with open(os.environ["NCCL_DEBUG_FILE"]) as f:
            nccl_debug_file_content = f.read()
            # if buffers were registered and symmetric kernels ran, NCCL_DEBUG
            # should show successful registration in debug output
            self.assertRegex(nccl_debug_file_content, "Symmetric")


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    @property
    def device(self):
        return f"cuda:{self.rank}"

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # No support for float16 for CPU tensors
        if device == torch.device("cpu"):
            half = torch.float32

        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # The tensors to pass to broadcast are identical to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{self.rank:d}")
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{self.rank:d}")
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        torch.distributed.all_reduce_coalesced(tensors, group=process_group)
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_manager_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{self.rank:d}")
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        with torch.distributed._coalescing_manager(
            group=process_group, device=device, async_ops=True
        ) as cm:
            for tensor in tensors:
                torch.distributed.all_reduce(tensor)
        self.assertEqual(len(cm.works), 1)
        cm.wait()
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @runOnRocmArch(MI300_ARCH)
    def test_intra_node_comm_all_reduce(self):
        from torch._C._distributed_c10d import _get_intra_node_comm_usage_counter
        from torch.testing._internal.common_cuda import SM80OrLater

        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            if not torch._C._cuda_canDeviceAccessPeer(self.rank, peer):
                raise SkipTest("Test requires p2p access")

        if not SM80OrLater:
            raise SkipTest("Test requires sm>=80")

        store = c10d.FileStore(self.file_name, self.world_size)
        os.environ["ENABLE_INTRA_NODE_COMM"] = "1"
        os.environ["TEST_INTRA_NODE_COMM"] = "1"
        torch.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        expect = self.world_size * (self.world_size - 1) // 2

        # IntraNodeComm currently only supports sum and bf16.
        # Verify that it is not used in the next two configurations.
        t = torch.full((4 * 1024 // 2,), self.rank).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 0)

        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.AVG)
        self.assertEqual(_get_intra_node_comm_usage_counter(), 0)

        # Verify that IntraNodeComm is used up to 10MB
        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 1)

        t = torch.full((512 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 2)

        t = torch.full((10 * 1024**2 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 3)

        # Verify that IntraNodeComm is not used beyond 10MB
        t = torch.full((10 * 1024**2 // 2 + 1,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 3)

        c10d.destroy_process_group()

    @requires_nccl()
    @requires_nccl_version(
        (2, 22), "Need NCCL 2.22+ for configuring estimate comm time"
    )
    @skip_if_lt_x_gpu(2)
    def test_time_estimate_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        torch.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{self.rank:d}")
        t = torch.full(
            (1024,),
            self.rank,
        ).cuda()
        with dist._time_estimator(group=process_group, device=device) as cm:
            c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(cm.estimated_time is not None)
        self.assertTrue(cm.estimated_time > 0)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_nccl(self):
        torch.cuda.set_device(self.rank)
        self._test_sequence_num_set_default_pg(backend="nccl")

    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_sequence_num_incremented_nccl_default(self):
        self._test_sequence_num_incremented_default_group("nccl")

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sequence_num_incremented_nccl_subgroup(self):
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_nccl_new_group(self):
        torch.cuda.set_device(self.rank)
        self._test_sequence_num_set_new_group(backend="nccl")

    def _test_pass_nccl_options(self, pg_opts):
        store = c10d.FileStore(self.file_name, self.world_size)
        # Test init_process_group accepts options
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=pg_opts,
        )

        # Test with new_group
        pg = c10d.new_group([0, 1], pg_options=pg_opts)
        # test the process group works as expected
        t = torch.tensor([self.rank + 1] * 10).cuda(self.rank)
        pg.allreduce(t).wait()
        expected_tensor = torch.tensor([3] * 10).cuda(self.rank)
        self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_nccl_options_high_priority_stream(self):
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.is_high_priority_stream = True
        self._test_pass_nccl_options(pg_opts)

    @requires_nccl()
    @requires_nccl_version(
        (2, 18), "Need NCCL 2.17+ for configuring NCCL communicators"
    )
    @skip_if_lt_x_gpu(2)
    def test_pass_nccl_options_config(self):
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.config.max_ctas = 4
        pg_opts.config.min_ctas = 2
        pg_opts.config.cga_cluster_size = 2
        pg_opts.config.net_name = "Socket"
        pg_opts.config.split_share = 1
        nccl_debug_file = tempfile.NamedTemporaryFile()
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_FILE"] = nccl_debug_file.name

        # Tests functionality when passing nccl config
        self._test_pass_nccl_options(pg_opts)

        # Tests if comms were configured
        nccl_debug_file_content = nccl_debug_file.read()
        max_ctas = re.search(rb"Max CTAs.*(\d+)|$", nccl_debug_file_content).group(1)
        min_ctas = re.search(rb"Min CTAs.*(\d+)|$", nccl_debug_file_content).group(1)
        split_share = re.search(
            rb"Split share.*(\d+)|$", nccl_debug_file_content
        ).group(1)
        cga_cluster_size = re.search(
            rb"CGA cluster.*(\d+)|$", nccl_debug_file_content
        ).group(1)
        net_name = re.search(
            rb"Using network.([a-zA-z]+)|$", nccl_debug_file_content
        ).group(1)
        self.assertEqual(pg_opts.config.max_ctas, int(max_ctas))
        self.assertEqual(pg_opts.config.min_ctas, int(min_ctas))
        self.assertEqual(pg_opts.config.cga_cluster_size, int(cga_cluster_size))
        self.assertEqual(pg_opts.config.net_name, net_name.decode())
        self.assertEqual(pg_opts.config.split_share, int(split_share))

        # Tests that config is inited correctly
        pg_opts = c10d.ProcessGroupNCCL.Options()
        nccl_cfg = c10d.ProcessGroupNCCL.NCCLConfig()
        self.assertEqual(pg_opts.config.min_ctas, -2147483648)
        self.assertEqual(nccl_cfg.min_ctas, -2147483648)

        # Tests that opts and config can be copied
        pg_opts_2 = copy.deepcopy(pg_opts)
        nccl_cfg_2 = copy.copy(pg_opts_2.config)
        pg_opts_2.config.min_ctas = 2
        nccl_cfg_2.min_ctas = 4
        self.assertEqual(pg_opts.config.min_ctas, -2147483648)
        self.assertEqual(pg_opts_2.config.min_ctas, 2)
        self.assertEqual(nccl_cfg_2.min_ctas, 4)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        c10d.all_reduce(t)
        expected_tensor = torch.tensor([3] * 10).cuda(2 * self.rank)
        self.assertEqual(expected_tensor, t)

        # Test with new_group
        pg = c10d.new_group([0, 1])
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        pg.allreduce(t).wait()
        self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([0])
        if self.rank == 0:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([1])
        if self.rank == 1:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        c10d.barrier(device_ids=[self.rank])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: Run collectives under context manager, and don't call wait on them.
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            input = torch.full(
                (10240, 10240), float(self.rank), device=f"cuda:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
            # Non-functional collectives run under the context manager is registered in the work registry.
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            # Running another collective on the same tensor should still work
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

        # Case 2: Run collectives not under context manager, and don't call wait on them.
        # NOTE: Here we intentionally test memory-stressed case.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)
        for _ in range(50000):
            input = torch.full(
                (1024, 1024), float(self.rank), device=f"cuda:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
        # Work registry size is unchanged, since non-functional collectives not run under
        # the context manager is not registered in the work registry.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        # Verify that c10d_functional.wait_tensor() can be invoked on
        # output tensor of non-functional collective
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: under context manager (i.e. work is registered in registry)
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            input1 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            torch.ops.c10d_functional.wait_tensor(input1)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

            input2 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            self.assertEqual(input1, input2)

        # Case 2: not under context manager (i.e. work is not registered in registry)
        input1 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        # this does not take effect, since the underlying wait_tensor() logic would not
        # be able to find the corresponding work object (because it's not registered in registry)
        torch.ops.c10d_functional.wait_tensor(input1)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

        input2 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work.wait()
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        self.assertEqual(input1, input2)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_nccl_warn_not_in_group_debug_detail(self):
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_nccl_warn_not_in_group_debug_info(self):
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_nccl_warn_not_in_group_debug_off(self):
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nncl_rank_membership(self):
        self._test_rank_membership(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_mismatch(self):
        self._test_tensor_dtype_mismatch(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_complex(self):
        self._test_tensor_dtype_complex(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensor = torch.zeros(2, dtype=torch.int64).to(self.rank)
        input_tensors = torch.arange(self.world_size * 2, dtype=torch.int64).to(
            self.rank
        )
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        dist.reduce_scatter_tensor(output_tensor, input_tensors)
        self.assertEqual(output_tensor, input_tensors[self.rank] * self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]
        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)


class NCCLTraceTestBase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["TORCH_NCCL_ENABLE_TIMING"] = (
            "0"  # see 'timing_enabled' parametrized tests
        )
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = self._trace_basename()
        os.environ["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] = self._trace_basename()
        self._spawn_processes()

    @classmethod
    def _run(
        cls,
        parent_conn,
        rank: int,
        test_name: str,
        file_name: str,
        parent_pipe,
        **kwargs,
    ) -> None:
        cls.parent = parent_conn
        super()._run(rank, test_name, file_name, parent_pipe)

    @property
    def local_device(self):
        return torch.device("cuda", self.rank_to_GPU[self.rank][0])

    def _join_processes(self, fn):
        # We need to patch sys.exit() as skip_if will use sys.exit() and
        # the exit code from the this process will not be caught.
        with mock.patch("sys.exit"):
            fn()
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self.children_pipes = []
        parent_pipes = []
        for _ in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)

        self._start_processes(wrap)

    def _create_process_group_nccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "nccl", world_size=self.world_size, rank=self.rank, store=store
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "nccl")

    def _trace_basename(self):
        # we pass the base to the env, and the dump util will append rank
        return os.path.join(self.tempdir.name, "trace_")

    def _trace_name(self, rank):
        return self._trace_basename() + str(rank)

    def started_or_scheduled(self, timing_enabled):
        return "started" if timing_enabled else "scheduled"


class NCCLTraceTest(NCCLTraceTestBase):
    def _verify_trace(self, t, include_collectives, timing_enabled, is_json):
        ver = t["version"]
        self.assertEqual(ver, "2.10")
        comm_lib_version = t["comm_lib_version"]
        torch_comm_lib_version = torch.cuda.nccl.version()
        self.assertEqual(
            comm_lib_version, ".".join(str(v) for v in torch_comm_lib_version)
        )
        pg_config = t["pg_config"]
        self.assertEqual(len(pg_config), 1)
        default_pg_info = pg_config["0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        pg_status = t["pg_status"]
        self.assertEqual(len(pg_status), 1)
        self.assertEqual(str(pg_status["0"]["last_enqueued_collective"]), "2")
        self.assertEqual(str(pg_status["0"]["last_completed_collective"]), "2")
        self.assertEqual(
            str(pg_status["0"]["last_started_collective"]),
            "2" if timing_enabled else "-1",
        )
        global_ranks = pg_config["0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        if include_collectives:
            self.assertEqual(len(t["entries"]), 2)
            t = t["entries"]
            last = t[-1]
            self.assertEqual(last["thread_id"], str(threading.current_thread().ident))
            self.assertEqual(last["thread_name"], "fr_test_thread")
            self.assertEqual(last["process_group"], ("0", "default_pg"))
            self.assertEqual(last["state"], "completed")
            s = last["time_discovered_started_ns"]
            f = last["time_discovered_completed_ns"]
            self.assertEqual(last["record_id"], 1)
            self.assertIsNotNone(f)
            if timing_enabled:
                self.assertIsNotNone(s)
                self.assertTrue(s <= f)
            # we don't collect stack traces in JSON at the moment
            if not is_json:
                self.assertIn("test_c10d_nccl.py", str(last["frames"]))
            self.assertEqual(last["input_sizes"], ((3, 4),))
            self.assertEqual(last["input_dtypes"], ["Float"])
            self.assertEqual(last["output_sizes"], ((3, 4),))
            self.assertEqual(last["output_dtypes"], ["Float"])
            self.assertEqual(last["collective_seq_id"], 2)
            self.assertEqual(last["timeout_ms"], 600000)
            now = datetime.now()
            event_created_time = datetime.fromtimestamp(
                last["time_created_ns"] / 1000000000
            )
            before_test = now - timedelta(minutes=1)
            self.assertTrue(before_test < event_created_time < now)
            if timing_enabled:
                # very loose bounds, measured 0.036 ms on devgpu
                self.assertTrue(0 < last["duration_ms"] < 100)
            else:
                self.assertTrue("duration_ms" not in last)
        else:
            self.assertTrue("entries" not in t)

    def load_libpthread_or_libc(self):
        import ctypes.util

        for base in ("pthread", "c"):
            path = ctypes.util.find_library(base)
            if path:
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        raise RuntimeError("Could not load pthread or libc")

    # Directly set thread name using threading.current_thread().name does not work
    # because we use pthread_getname_np to get the thread’s OS-level name in C++
    def set_thread_name(self, name):
        import ctypes

        lib = self.load_libpthread_or_libc()
        pthread_self = lib.pthread_self
        pthread_self.restype = ctypes.c_void_p
        pthread_setname_np = lib.pthread_setname_np
        pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        # Get current pthread handle
        tid = pthread_self()

        # Set name
        pthread_setname_np(tid, name.encode())

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("include_collectives", [True, False])
    def test_short_json(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = json.loads(
            torch._C._distributed_c10d._dump_nccl_trace_json(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(t, include_collectives, timing_enabled, True)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("include_collectives", [True, False])
    def test_short_pickle(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = pickle.loads(
            torch._C._distributed_c10d._dump_nccl_trace(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(
            t,
            include_collectives=include_collectives,
            timing_enabled=timing_enabled,
            is_json=True,
        )
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_dump_pipe(self):
        def open_file_with_timeout(file_path, mode, timeout=1.0):
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(file_path):
                    return open(file_path, mode)
                time.sleep(0.1)
            raise FileNotFoundError

        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")

            dump_file = self._trace_name(rank=0)
            pipe_file = dump_file + ".pipe"
            with open_file_with_timeout(pipe_file, "w") as f:
                f.write("1\n")
            with open_file_with_timeout(dump_file, "rb", timeout=10.0) as f:
                self.assertTrue("all_reduce" in str(pickle.load(f)))

            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        self.parent.send("next")
        self.parent.recv()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_long(self):
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            # test some other primitives to make sure
            # their strings are valid
            xs = [torch.ones(3, 4, device=device)]
            pg.broadcast(xs).wait()
            pg.allreduce(xs).wait()
            pg.reduce(xs).wait()
            ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
            pg.allgather(ys, xs).wait()
            pg.reduce_scatter(xs, ys).wait()
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        self.assertEqual(last["profiling_name"], "nccl:all_reduce")
        self.assertEqual(last["state"], "completed")
        self.assertIn("test_c10d_nccl.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["timeout_ms"], 600000)
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_barrier_profiling(self):
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        f = pg.barrier()
        f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 2)
        first = t[0]
        last = t[-1]
        self.assertEqual(first["profiling_name"], "nccl:all_reduce_barrier")
        self.assertEqual(last["profiling_name"], "nccl:all_reduce")
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_trace_while_all_works_retired(self):
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        # send more works than the buffer size to overwrite the previous entry
        for _ in range(12):
            a = [torch.ones(3, 4, device=device)]
            pg.broadcast(a).wait()
        torch.cuda.synchronize(device=device)

        # wait for all works to be retired
        pg._wait_for_pending_works()
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        last = t[-1]
        self.assertEqual(last["retired"], True)
        self.assertEqual(last["state"], "completed")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("only_active", [True, False])
    def test_trace_while_active(self, timing_enabled, only_active):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.cuda.Event()
            e.record()
            if self.rank != 0:
                pg.allreduce(a).wait()
            e.synchronize()
            t = pickle.loads(
                torch._C._distributed_c10d._dump_nccl_trace(onlyActive=only_active)
            )
            t = t["entries"]
            if only_active:
                if self.rank == 0:
                    self.assertEqual(len(t), 0)
                else:
                    self.assertEqual(len(t), 1)
            if not only_active:
                if self.rank == 0:
                    self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                    self.assertEqual(t[-1]["collective_seq_id"], 2)

                    # ROCm runtime used to call uSleep(20 µs)inside the default‑signal busy-wait loop.
                    # Now, this sleep is removed which lets the host thread spin continuously
                    # Therefore, the state can either be scheduled or started before test dumps the trace.
                    if (
                        torch.version.hip
                        and _get_torch_rocm_version() >= (6, 4)
                        and timing_enabled
                    ):
                        assert t[-1]["state"] in ("scheduled", "started")
                    else:
                        self.assertEqual(
                            t[-1]["state"], self.started_or_scheduled(timing_enabled)
                        )

            self.parent.send("next")
            self.assertEqual("next", self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.cuda.synchronize(device=device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    def test_trace_while_stuck(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.cuda.Event()
            e.record()

            def gather_trace():
                e.synchronize()
                # give the other thread some time to fill the cuda buffer
                time.sleep(5)
                t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
                t = t["entries"]
                self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                if self.rank == 0:
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )
                    self.assertIsNone(t[-1]["time_discovered_completed_ns"])
                # this will eventually cause the missing rank 0
                # to continue which will unblock the non-zero ranks
                self.parent.send("next")

            if self.rank != 0:
                pg.allreduce(a).wait()
                th = threading.Thread(target=gather_trace)
                th.start()
                # fill the cuda buffer, at around 1024 events
                # this will stall
                for _ in range(2000):
                    a = a + a
                th.join()
            else:
                gather_trace()

            self.assertEqual("next", self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.cuda.synchronize(device=device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "op_sizes_per_coalesce",
        [
            [(2, 3)],
            [(2, 3), (5, 5), (1,)],
        ],
    )
    @parametrize("timing_enabled", [True, False])
    def test_batched_send_recv(self, op_sizes_per_coalesce, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's cuda events
        """

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        num_coalesced_ops = 20
        ops_per_coalesce = len(op_sizes_per_coalesce)
        for _ in range(num_coalesced_ops):
            ops = []
            for input_sizes in op_sizes_per_coalesce:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    ops.append(dist.P2POp(dist.irecv, tensor, 1))
                elif self.rank == 1:
                    tensor *= 2
                    ops.append(dist.P2POp(dist.isend, tensor, 0))

            dist.batch_isend_irecv(ops).pop().wait()

        torch.cuda.synchronize(device=self.local_device)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t["entries"]), num_coalesced_ops * (ops_per_coalesce + 1))

        expected_record_id = 0
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_coalesced_ops):
            first_op = seq * (ops_per_coalesce + 1)
            coalesced_op = first_op + ops_per_coalesce
            for p2p_op_idx, input_sizes in zip(
                range(first_op, coalesced_op, 1), op_sizes_per_coalesce
            ):
                # the indivudal ops inside the coalescing group the individual op metadata,
                # but not the timing info coming from the actual coalesced kernel
                profiling_name = (
                    "nccl:recv 0<-1" if self.rank == 0 else "nccl:send 1->0"
                )
                self.assertEqual(
                    t["entries"][p2p_op_idx]["record_id"], expected_record_id
                )
                expected_record_id += 1
                self.assertEqual(
                    t["entries"][p2p_op_idx]["profiling_name"], profiling_name
                )
                # we don't increment collective_seq_id for p2p ops.
                self.assertEqual(t["entries"][p2p_op_idx]["collective_seq_id"], 0)
                self.assertEqual(t["entries"][p2p_op_idx]["p2p_seq_id"], expected_seq)
                self.assertEqual(t["entries"][p2p_op_idx]["op_id"], expected_op_id)
                expected_op_id += 1
                self.assertEqual(t["entries"][p2p_op_idx]["input_sizes"], [input_sizes])
                self.assertEqual(
                    t["entries"][p2p_op_idx]["output_sizes"], [input_sizes]
                )
                # duration doesn't get tagged onto individual ops yet, nor is their state updated
                self.assertEqual(t["entries"][p2p_op_idx]["state"], "scheduled")
                self.assertTrue("duration_ms" not in t["entries"][p2p_op_idx])

            # the coalesced op has no metadata but indicates that coalescing was used,
            # and accurately reflects the timing and state info for the whole group
            self.assertEqual(
                t["entries"][coalesced_op]["record_id"], expected_record_id
            )
            expected_record_id += 1
            self.assertEqual(
                t["entries"][coalesced_op]["profiling_name"], "nccl:coalesced"
            )
            self.assertEqual(t["entries"][coalesced_op]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][coalesced_op]["state"], "completed")
            self.assertEqual(t["entries"][coalesced_op]["input_sizes"], [])
            self.assertEqual(t["entries"][coalesced_op]["output_sizes"], [])
            if timing_enabled:
                duration = t["entries"][coalesced_op]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][coalesced_op])
            self.assertEqual(t["entries"][coalesced_op]["timeout_ms"], 600000)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "op_sizes",
        [
            [(2, 3)],
            [(2, 3), (5, 5), (1,)],
        ],
    )
    @parametrize("timing_enabled", [True, False])
    def test_individual_send_recv(self, op_sizes, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's cuda events
        """

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        num_repeats = 10
        ops_per_repeat = len(op_sizes)
        for _ in range(num_repeats):
            for input_sizes in op_sizes:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    dist.recv(tensor, 1)
                elif self.rank == 1:
                    tensor *= 2
                    dist.send(tensor, 0)

        torch.cuda.synchronize(device=self.local_device)
        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t["entries"]), num_repeats * (ops_per_repeat))
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_repeats * ops_per_repeat):
            input_sizes = op_sizes[seq % ops_per_repeat]
            profiling_name = "nccl:recv 0<-1" if self.rank == 0 else "nccl:send 1->0"
            self.assertEqual(t["entries"][seq]["profiling_name"], profiling_name)
            # we don't increment collective_seq_id for p2p ops.
            self.assertEqual(t["entries"][seq]["collective_seq_id"], 0)
            self.assertEqual(t["entries"][seq]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][seq]["op_id"], expected_op_id)
            expected_op_id += 1
            self.assertEqual(t["entries"][seq]["input_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["output_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["state"], "completed")

            if timing_enabled:
                duration = t["entries"][seq]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][seq])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_allgather_uneven(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        output_split_sizes = [i + 1 for i in range(self.world_size)]
        sum_len = sum(output_split_sizes)
        output_tensor = torch.zeros(sum_len, 2).to(self.rank)
        expected_tensor = torch.ones(sum_len, 2).to(self.rank)
        input_tensor = torch.ones(output_split_sizes[self.rank], 2).to(self.rank)

        dist.all_gather(
            list(torch.split(output_tensor, output_split_sizes)), input_tensor
        )
        torch.cuda.synchronize(device=self.rank)
        self.assertEqual(output_tensor, expected_tensor)
        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t["entries"]), self.world_size + 1)
        for i in range(self.world_size):
            self.assertEqual(t["entries"][i]["profiling_name"], "nccl:_broadcast_oop")
            # collective_seq_id should be incremented once.
            self.assertEqual(t["entries"][i]["collective_seq_id"], 1)
            self.assertEqual(t["entries"][i]["input_sizes"], [[i + 1, 2]])
            self.assertEqual(
                t["entries"][i]["output_sizes"],
                [[i + 1, 2]],
            )
            self.assertEqual(t["entries"][i]["state"], "scheduled")
            # No event is recorded for individual ops
            self.assertTrue("time_discovered_completed_ns" in t["entries"][i])
        self.assertEqual(
            t["entries"][self.world_size]["profiling_name"], "nccl:ALLGATHER_coalesced"
        )

    # TODO(whc) test out other ops (And combinations of ops, if that's valid?)
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_coalescing_manager_collective(self, timing_enabled):
        """
        The coalescing manager api works by accumulating operations in python via a contextmanager, and then making
        one call into c++ to an <op>_coalesced API.  It has limited support for ops and has been added recently to
        avoid overheads of making individual py-cpp calls.  This complicates flight recording..

        For now, flight recording of coalescing_manager collectives is less detailed than cpp coalesced collectives.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]

        # TODO(whc) make this work with bigger world or something
        self.assertEqual(self.world_size, 2, self.world_size)

        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

        torch.cuda.synchronize(device=self.rank)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())

        self.assertEqual(
            len(t["entries"]), 1
        )  # one for the reduce_scatter_tensor_coalesced
        self.assertEqual(
            t["entries"][0]["profiling_name"], "nccl:reduce_scatter_tensor_coalesced"
        )
        # collective_seq_id should be incremented once.
        self.assertEqual(t["entries"][0]["collective_seq_id"], 1)
        self.assertEqual(t["entries"][0]["input_sizes"], [[2, 2], [2, 2]])
        self.assertEqual(
            t["entries"][0]["output_sizes"],
            [
                [
                    2,
                ],
                [
                    2,
                ],
            ],
        )
        self.assertEqual(t["entries"][0]["state"], "completed")
        if timing_enabled:
            duration = t["entries"][0]["duration_ms"]
            self.assertTrue(0.001 < duration < 10000, duration)
        else:
            self.assertTrue("duration_ms" not in t["entries"][0])


def check_if_test_is_skipped(fn):
    def wrapper(self, *args, **kwargs):
        for skip in TEST_SKIPS.values():
            if self.processes[0].exitcode == skip.exit_code:
                return MultiProcessTestCase._check_return_codes(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


class NCCLTraceTestDumpOnTimeoutBase(NCCLTraceTestBase):
    timeout_sec = 1

    def _create_process_group_nccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=NCCLTraceTestDumpOnTimeoutBase.timeout_sec),
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 0)

    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None


@skip_but_pass_in_sandcastle
class NCCLTraceTestDumpOnTimeout(NCCLTraceTestDumpOnTimeoutBase):
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_timeout_dumps(self, timing_enabled):
        # dump on heartbeatmonitor thread
        os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"] = "1000"
        # need rank0 to crash before looking for its output file
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for rank0 to crash before looking for its output file
            # we rely on rank0 holding off its abort long enough to dump the debug info
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
                self.assertEqual(t[1]["collective_seq_id"], 2)
                self.assertEqual(
                    t[1]["state"], self.started_or_scheduled(timing_enabled)
                )

            self.assertFalse(os.path.exists(self._trace_name(rank=1)))

            return

        pg = self._create_process_group_nccl()
        if timing_enabled:
            # we force disabled timing in setup, since there is no 'disable' function
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will crash before it passes the sync, but rank1 will exit quickly and cleanly
            torch.cuda.synchronize(device=device)


instantiate_parametrized_tests(ProcessGroupNCCLGroupTest)
instantiate_parametrized_tests(NCCLTraceTestDumpOnTimeout)
instantiate_parametrized_tests(NCCLTraceTest)


@skip_but_pass_in_sandcastle
class NCCLTraceTestTimeoutDumpOnStuckRanks(NCCLTraceTestDumpOnTimeoutBase):
    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, -6)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_dumps_on_stuck_ranks(self):
        # need rank0 to crash quicker after detecting timeout
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1"
        # restore this env var to its prior default in case another test changed it
        os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"] = "1000"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for both ranks' output
            # file, and we rely on rank1 to sleep long enough to dump the debug info.
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), -6)
            self.assertTrue(os.path.exists(self._trace_name(rank=1)))
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
            with open(self._trace_name(rank=1), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 1)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
            return

        pg = self._create_process_group_nccl()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will get stuck, timeout and then signal a timeout to all ranks.
            torch.cuda.synchronize(device=device)

            if self.rank == 1:
                # Force rank 1 to idle so that it will eventually timeout as well after
                # getting the global signal to dump the debugging info.
                time.sleep(600)


@skip_but_pass_in_sandcastle
class NcclErrorDumpTest(NCCLTraceTestBase):
    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort with exception and rank1 to exit with exit 1
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 1)

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_dump(self):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        # need rank0 to dump before abort
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "5"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for dump
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), 1)
            # verify that the trace file exists for rank0
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            return

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            # expect an error to be raised
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # Block the current stream on the NCCL stream
                work.wait()
                # Run some GPU operations
                torch.rand(10).cuda(self.rank)
        elif self.rank == 1:
            # Clean up structures (ex: files for FileStore before going down)
            del process_group
            sys.exit(1)


# tests that needs to be run with a larger world size
class ProcessGroupNCCLLargerScaleTest(MultiProcessTestCase):
    def _create_process_group_nccl(self, store, opts, device_id=None):
        # create nccl processgroup with opts
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 8

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "nccl")

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_if_lt_x_gpu(8)
    def test_comm_split_group_larger_scale(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        ng1 = c10d.split_group(pg, [[0, 1], [2, 3, 4, 5, 6, 7]])

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        # dist.broadcast take Source rank on global process group
        if self.rank < 2:
            dist.broadcast(tensor, 0, group=ng1)
            self.assertEqual(tensor, torch.full((1,), 0))
        else:
            dist.broadcast(tensor, 2, group=ng1)
            self.assertEqual(tensor, torch.full((1,), 2))

        # test split with only one colored group, other ranks should be no color split.
        ng2 = c10d.split_group(pg, [[5, 6, 7]])
        self.assertEqual(backend.comm_split_count(), 2)

        if self.rank >= 5:
            tensor2 = torch.full((1,), self.rank).cuda(device)
            dist.broadcast(tensor2, 7, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 7))
        else:
            self.assertEqual(ng2, None)
        # a barrier and a cuda sync before destroying all pgs.
        dist.barrier(pg)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_if_lt_x_gpu(8)
    def test_comm_recursive_split_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        # split the default PG into 2 subgroups, each subgroup (ng1) has 4 ranks.
        tensor1 = torch.full((1,), self.rank).cuda(device)
        ng1 = c10d.split_group(pg, [[0, 1, 2, 3], [4, 5, 6, 7]])
        backend1 = ng1._get_backend(torch.device(device))
        if self.rank < 4:
            dist.broadcast(tensor1, 0, group=ng1)
            self.assertEqual(tensor1, torch.full((1,), 0))
        else:
            dist.broadcast(tensor1, 4, group=ng1)
            self.assertEqual(tensor1, torch.full((1,), 4))

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(backend1.comm_split_count(), 0)

        # further split ng1 into 2 subgroups, each subgroup (ng2) has 2 ranks.
        tensor2 = torch.full((1,), self.rank).cuda(device)
        ng2 = c10d.split_group(ng1, [[0, 1], [2, 3]])
        backend2 = ng2._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(backend1.comm_split_count(), 1)
        self.assertEqual(backend2.comm_split_count(), 0)

        # execute collective calls within each 2-rank pg
        if self.rank == 0 or self.rank == 1:
            dist.broadcast(tensor2, 1, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 1))

        if self.rank == 2 or self.rank == 3:
            dist.broadcast(tensor2, 2, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 2))

        if self.rank == 4 or self.rank == 5:
            dist.broadcast(tensor2, 5, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 5))

        if self.rank == 6 or self.rank == 7:
            dist.broadcast(tensor2, 6, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 6))
        # a barrier and a cuda sync before destroying all pgs.
        dist.barrier(pg)
        torch.cuda.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    assert not torch.cuda._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()
