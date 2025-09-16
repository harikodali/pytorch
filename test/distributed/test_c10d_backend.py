import copy
import os
import random
import sys
from contextlib import contextmanager
from datetime import timedelta
from enum import auto, Enum
from itertools import chain, product
from unittest import mock

import torch
import torch.distributed as c10d


if not c10d.is_available(): # or not c10d.is_nccl_available():
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
from torch._C._distributed_c10d import OpType
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    HAS_ACCELERATOR,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
    sm_is_or_higher_than,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    retry_on_connect_failures,
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
BFLOAT16_AVAILABLE = True #torch.accelerator.is_available() and (
    # torch.version.cuda is not None or torch.version.hip is not None
# )

DEVICE_TYPE = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
BACKEND = dist.get_default_backend_for_device(DEVICE_TYPE)

class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    @requires_accelerator_dist_backend()
    @skip_but_pass_in_sandcastle_if(not HAS_ACCELERATOR, "No GPUs available, skipping test")
    def test_common_errors(self):
        vars = {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(common.find_free_port()),
        }

        class Env:
            def __init__(self, vars):
                self.env_patcher = mock.patch.dict(os.environ, vars, clear=True)

            def __enter__(self):
                self.env_patcher.start()

            def __exit__(self, type, value, traceback):
                self.env_patcher.stop()

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend=BACKEND, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend=BACKEND, rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            c10d.init_process_group(backend=BACKEND, rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend=BACKEND)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "MASTER_ADDR")):
            self.assertEqual(None, os.environ.get("MASTER_ADDR"))
            with self.assertRaisesRegex(ValueError, "MASTER_ADDR expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "MASTER_PORT")):
            self.assertEqual(None, os.environ.get("MASTER_PORT"))
            with self.assertRaisesRegex(ValueError, "MASTER_PORT expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?world_size={1}")
            _, _, size = next(gen)
            self.assertEqual(size, 1)

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            gen = c10d.rendezvous(f"env://?rank={0}")
            _, rank, _ = next(gen)
            self.assertEqual(rank, 0)

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?rank={0}&world_size={1}")
            _, rank, size = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    @requires_accelerator_dist_backend()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(not HAS_ACCELERATOR, "No GPUs available, skipping test")
    def test_default_store_timeout_nccl(self):
        self._test_default_store_timeout(BACKEND)


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_nccl_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_propagate_error_reason(self):
        # Need to use TORCH_NCCL_BLOCKING_WAIT and not ASYNC_ERROR_HANDLING,
        # otherwise process will be taken down and we can't check for errors.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        # Need to disable TORCH_NCCL_DUMP_ON_TIMEOUT otherwise this test times out
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        # provide sufficient timeout to initialize NCCL comm.
        pg = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=15)
        )
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        pg.barrier().wait(timedelta(seconds=5))
        # Simulate stuckness in rank 0.
        if self.rank == 0:
            pg_gloo.barrier().wait()
        inp = torch.ones(1).to(self.rank)

        if self.rank != 0:
            # Time out due to rank 0 not calling into allreduce.
            with self.assertRaises(dist.DistBackendError):
                pg.allreduce([inp]).wait(timedelta(seconds=5))

            # Now when nonzero rank attempts to use communicator, original failure reason should be logged.
            try:
                pg.allreduce([torch.ones(2).to(self.rank)]).wait()
            except dist.DistBackendError as e:
                self.assertTrue("aborted" in str(e))
            else:
                self.fail("Expected error to be raised!")

            # Unblock rank 0
            pg_gloo.barrier().wait()

        # TODO: We can also test that if rank 0 attempts to use the communicator,
        # then we should error out with the info that it was aborted due to
        # timeout on another rank. Although this would only be the case after
        # the watchdog has run on the rank, and there is no reliable way
        # to confirm it has run.

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_multi_device_ids_not_allowed(self):
        int_devices = list(range(torch.accelerator.device_count()))
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            self._test_nccl_backend(devices, int_devices)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_device_ids_None(self):
        self._test_nccl_backend(None, None)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_empty_device_ids(self):
        # This tests the backward compatibility of accepting an empty list as `device_ids`,
        # although we no longer document this in favor of the default value of `None`,
        # which is consistent with multi-device modules and CPU modules.
        self._test_nccl_backend(None, [])

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_multi_device_module_device_ids_None(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, int_devices)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, devices)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(8)
    def test_nccl_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_ddp_multi_device_module_config(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]

        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        process_group = self._get_process_group()

        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)

        with self.assertRaisesRegex(
            ValueError,
            "DistributedDataParallel device_ids and output_device arguments only work with "
            "single-device/multiple-device GPU modules or CPU modules",
        ):
            DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

        with self.assertRaisesRegex(
            ValueError, "input module must be on the same type of devices"
        ):
            model.fc1 = model.fc1.cpu()
            DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

    def _test_fp16(self, gradient_as_bucket_view=False):
        process_group = self._get_process_group()

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).to(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.tensor([[2**15]]).to(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        self._test_fp16()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16_grad_is_view(self):
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class ForwardReturnValueModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # The first softmax does NOT include fc3 in its autograd graph
                # whereas the second softmax DOES. If we pass only the first
                # tensor we see in the output to the reducer, it marks the
                # gradient for fc3 as ready (because it doesn't show up). If
                # downstream uses of this return value choose to differentiate
                # against the second output tensor, it would still receive a
                # gradient and a callback for this tensor, resulting in a crash.
                return fn(
                    F.softmax(x, dim=1),
                    F.softmax(self.fc3(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            ForwardReturnValueModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Always run "backward" to ensure the reducer is called by autograd.
        # If we don't correctly capture the output tensors from the return value,
        # the reducer won't see a hook for the unused parameter, and throw an error.
        # The correct capture is what we're testing in this function.
        def test(box, unbox):
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()

        # Test with identity return value
        test(
            box=lambda x, y: (x, y),
            unbox=lambda obj: obj[1],
        )

        # Test with list return value
        test(
            box=lambda x, y: ["foo", x, "bar", y],
            unbox=lambda obj: obj[3],
        )

        # Test with tuple return value
        test(
            box=lambda x, y: ("foo", x, "bar", y),
            unbox=lambda obj: obj[3],
        )

        # Test with dict return value
        test(
            box=lambda x, y: {"foo": "bar", "a": x, "b": y},
            unbox=lambda obj: obj["b"],
        )

        # Test with list with dict return value
        test(
            box=lambda x, y: ["foo", "bar", {"a": x, "b": y}],
            unbox=lambda obj: obj[2]["b"],
        )

        # Test with dict with list return value
        test(
            box=lambda x, y: {"foo": "bar", "list": [0, x, 1, y]},
            unbox=lambda obj: obj["list"][3],
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_with_lazy_parameters(self):
        process_group = self._get_process_group()
        with self.assertRaisesRegex(
            RuntimeError, "Modules with uninitialized parameters"
        ):
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        torch.accelerator.set_device_index(self.rank)
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )
        process_group = c10d.distributed_c10d._get_default_group()

        class FindUnusedParametersModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # Return the fc3 module so that the caller can invoke it
                # outside of the forward function. While this is bad practice,
                # we can use it to trigger a reducer error.
                return (F.softmax(x, dim=1), self.fc3)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        ddp_model = None

        def test_find_unused_parameters(
            find_unused_parameters, test_default=False, gradient_as_bucket_view=False
        ):
            if test_default:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            else:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    find_unused_parameters=find_unused_parameters,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            nonlocal ddp_model
            ddp_model = model

            output, fc3 = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        # First test that finding unused params under these conditions is to
        # trigger an error when `backward` is called (because fc3 is an unused
        # parameter and will therefore be marked ready twice).
        try:
            test_find_unused_parameters(
                True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.assertTrue(
                str(ex).startswith(
                    "Expected to mark a variable ready only once.",
                )
            )
            unused_index = 2
            unused_index_str = f"Parameter at index {unused_index}"
            model = ddp_model.module
            for module_name, module in model.named_modules():
                if module == model.fc3:
                    for parameter_name, _ in module.named_parameters(recurse=False):
                        unused_fqn = f"{module_name}.{parameter_name}"
                        # Only one such parameter in model.fc3, since bias=False
                        break

            if dist.get_debug_level() != dist.DebugLevel.OFF:
                unused_index_str += f" with name {unused_fqn}"

            self.assertTrue(unused_index_str in str(ex))
        else:
            self.fail("Expected exception")

        dist.barrier(process_group)

        # Then test that the default behavior can be overridden by setting
        # `find_unused_parameters=False`.
        try:
            test_find_unused_parameters(
                False, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

        # Test find_unused_parameters defaults to False
        try:
            test_find_unused_parameters(
                True, test_default=True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

    # TODO: Combine the following tests once https://github.com/pytorch/pytorch/issues/55967
    # is resolved.
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_debug_detail(self):
        self._test_find_unused_parameters_kwarg()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_debug_info(self):
        self._test_find_unused_parameters_kwarg()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_debug_off(self):
        self._test_find_unused_parameters_kwarg()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_off(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class MultipleOutputModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Compute loss and gradients for both outputs
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class NoGradModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            NoGradModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)

        # After initialization, no parameter has their gradient set.
        check_no_grads()

        # Run `forward` function with torch.no_grad()
        with torch.no_grad():
            output = model(input)
            self.assertTrue(isinstance(output, torch.Tensor))

        # No parameter should have their gradient set.
        check_no_grads()

    def _test_accumulate_gradients_module(self, gradient_as_bucket_view=False):
        # This is NOT the recommended way to implement accumulating grads, but
        # we would like to make sure DDP does not mess up with the underlying
        # module.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device(f"{DEVICE_TYPE}:" + str(i)) for i in int_devices]
        process_group = self._get_process_group()
        global_batch_size = self.world_size

        model, ddp_model, input, target = self._prepare_single_device_module(
            process_group, devices, devices, global_batch_size, gradient_as_bucket_view
        )

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        # ensure accumulate grads works with no_grad
        with torch.no_grad():
            ddp_model.train()
            ddp_model.module(input)

        # Check two model parameters over 4 iterations.
        # Use 4 iterations because we alternate between reducing and
        # not reducing and want to make sure we switch both ways.
        for iteration in range(4):
            step_model(model, input, target)

            if iteration % 2 == 0:
                # Skip gradients sync without calling prepare_for_backward
                step_model(
                    ddp_model.module,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertNotEqual(i.grad, j.grad)
            else:
                step_model(
                    ddp_model,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module(self):
        self._test_accumulate_gradients_module()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module_with_grad_is_view(self):
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_failure_recovery(self):
        process_group = self._get_process_group()

        # need to create a separate file for the recovered FileStore, because
        # the original one will be deleted when destructing the first FileStore.
        recovery_filename = self.file_name + "_recovery"

        if self.rank == 0:
            # the file will be deleted by the recovered FileStore
            open(recovery_filename, "w").close()

        # not necessary to run barrier here, as DDP will synchronize

        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = TestModel().float().to(device_id)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

        del ddp
        c10d.destroy_process_group(process_group)

        store = c10d.FileStore(recovery_filename, self.world_size)
        c10d.init_process_group(
            BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        dist.init_process_group(
            BACKEND,
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        self.assertFalse(dist.is_initialized())

    def _test_grad_layout(self, replica_devices, layer_devs, local_batch_size):
        process_group = self._get_process_group()

        global_batch_size = local_batch_size * self.world_size

        # Carry out some trials with small buckets and some with big buckets.
        bucketsizes = (0.000001, 25)
        # Tuples of lists.  Each list describes per-layer characteristics for one trial.
        layer_formats = (
            [torch.contiguous_format] * 4,
            [torch.channels_last] * 2 + [torch.contiguous_format] * 2,
            [torch.channels_last] * 4,
        )
        layer_dtypes = (
            [torch.float] * 4,
            [torch.float] * 2 + [torch.half] * 2,
            [torch.half] * 4,
        )

        input_dev = layer_devs[0] if isinstance(layer_devs, list) else layer_devs
        target_dev = layer_devs[-1] if isinstance(layer_devs, list) else layer_devs
        input = torch.randn(
            (global_batch_size, 8, 8, 8), device=input_dev, dtype=torch.float
        )
        target = torch.randn(
            (global_batch_size, 8, 4, 4), device=target_dev, dtype=torch.float
        )
        local_batch_start = self.rank * local_batch_size
        local_batch_end = (self.rank + 1) * local_batch_size

        # Reducer.cpp sneakily creates one "initial bucket" that ignores the "bucket_cap_mb"
        # argument.  The following makes sure the initial bucket also complies.
        @contextmanager
        def first_bucket_size(ddp_bucket_mb):
            old_DEFAULT_FIRST_BUCKET_BYTES = dist._DEFAULT_FIRST_BUCKET_BYTES
            dist._DEFAULT_FIRST_BUCKET_BYTES = int(ddp_bucket_mb * 1.0e6)
            try:
                yield
            finally:
                dist._DEFAULT_FIRST_BUCKET_BYTES = old_DEFAULT_FIRST_BUCKET_BYTES

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=False
        ):
            for formats, dtypes, bucketsize in product(
                layer_formats, layer_dtypes, bucketsizes
            ):
                with first_bucket_size(bucketsize):
                    model_msg = f"rank = {self.rank} formats = {formats} dtypes = {dtypes} bucketsize = {bucketsize} "
                    try:
                        m = ConvNet(layer_devs, formats, dtypes)
                        m_ddp = DistributedDataParallel(
                            copy.deepcopy(m),
                            device_ids=replica_devices,
                            process_group=process_group,
                            bucket_cap_mb=bucketsize,
                        )
                        opt = torch.optim.SGD(m.parameters(), lr=0.1)
                        opt_ddp = torch.optim.SGD(m_ddp.parameters(), lr=0.1)
                        has_half = any(p.dtype is torch.half for p in m.parameters())
                        tol = 1.0e-3 if has_half else 1.0e-5
                    except BaseException:
                        # Prints case-specific debugging info to narrow down failing case.
                        print(
                            "Caught exception during model creation for " + model_msg,
                            flush=True,
                        )
                        raise
                    # 3 iters:  First iter creates grads, second iter retests after rebucketing,
                    # third iter tries zeroed grads.
                    for it in range(3):
                        iter_msg = f"iter = {it} " + model_msg
                        named_msg = iter_msg
                        try:
                            F.mse_loss(m(input).float(), target).backward()
                            F.mse_loss(
                                m_ddp(input[local_batch_start:local_batch_end]).float(),
                                target[local_batch_start:local_batch_end],
                            ).backward()
                            for i, ((layer_name, m_child), m_ddp_child) in enumerate(
                                zip(m.named_children(), m_ddp.module.children())
                            ):
                                named_msg = layer_name + ".weight" + " " + iter_msg
                                self.assertTrue(
                                    m_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                self.assertTrue(
                                    m_ddp_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                for (param_name, p), p_ddp in zip(
                                    m_child.named_parameters(),
                                    m_ddp_child.parameters(),
                                ):
                                    named_msg = (
                                        layer_name + "." + param_name + " " + iter_msg
                                    )
                                    self.assertEqual(
                                        p.grad, p_ddp.grad, rtol=tol, atol=tol
                                    )
                            opt.step()
                            opt_ddp.step()
                            if it == 0:
                                for p, p_ddp in zip(m.parameters(), m_ddp.parameters()):
                                    p.grad = None
                                    p_ddp.grad = None
                            else:
                                m.zero_grad()
                                m_ddp.zero_grad()
                        except BaseException:
                            # Makes sure we still get info if an error occurred somewhere other than the asserts.
                            print(
                                "Caught exception during iterations at " + named_msg,
                                flush=True,
                            )
                            raise

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        dev0 = torch.device(f"{DEVICE_TYPE}:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # Tells DDP to use just one device.
        replica_devices = [dev0]
        # Tells _test_grad_layout to construct ConvNet with all layers on this process's first assigned device.
        layer_devs = dev0
        local_batch_size = 16
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @skip_if_rocm_multiprocess
    def test_grad_layout_2devicemodule(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device(f"{DEVICE_TYPE}:" + str(int_devices[0]))
        dev1 = torch.device(f"{DEVICE_TYPE}:" + str(int_devices[1]))
        # DDP's default behavior for a multi-device module is "don't replicate."
        replica_devices = None
        # Tells _test_grad_layout to constructs this process's ConvNet on 2 devices, with 2 layers on each device.
        layer_devs = [dev0] * 2 + [dev1] * 2
        local_batch_size = 16
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_param_layout_mismatch_error(self):
        process_group = self._get_process_group()

        dev0 = torch.device(f"{DEVICE_TYPE}:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        layer_devs = dev0
        layer_formats = (
            [torch.contiguous_format] * 4
            if self.rank == 0
            else [torch.channels_last] * 4
        )
        layer_dtypes = [torch.float] * 4

        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        if self.rank == 0:
            DistributedDataParallel(m, device_ids=[dev0], process_group=process_group)
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                ".* appears not to match strides of the same param in process 0",
            ):
                DistributedDataParallel(
                    m, device_ids=[dev0], process_group=process_group
                )

    def _gpu_model_with_ddp_comm_hook(
        self,
        process_group,
        hook=None,
        gradient_as_bucket_view=False,
        state=None,
        static_graph=False,
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_nccl(self):
        """
        This unit test verifies whether the Future object is passed properly using nccl backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook_nccl(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        """
        This unit test verifies whether a DDP communication hook that just calls
        allreduce gives the same result with the case of no hook registered.
        Without the then callback, the future_value in reducer is no longer
        a PyObject, and this unit test verifies future_value is properly checked.
        """
        process_group = self._get_process_group()

        def allreduce_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            return (
                process_group.allreduce(tensors)
                .get_future()
                .then(lambda fut: fut.value()[0])
            )

        # Get GPU model with allreduce_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_hook, gradient_as_bucket_view, static_graph
        )

        # check whether the grads are equal to what DDP without hook would return.
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_default_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE, FP16_COMPRESS
        and BF16_COMPRESS, can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # For these default DDP comm hooks, the only state is process group.
        state = process_group
        hook_options = [default.allreduce_hook, default.fp16_compress_hook]
        if (
            not TEST_WITH_ROCM
            and BFLOAT16_AVAILABLE
            # and c10d.is_nccl_available()
            # and torch.accelerator.nccl.version() >= (2, 10)
        ):
            hook_options.append(default.bf16_compress_hook)
        for hook in hook_options:
            # Get GPU model with the hook registered.
            # The first arg 'process_group' is used for initializing the test environment,
            # so it cannot be replaced by 'state', although they have the same value.
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group, hook, gradient_as_bucket_view, state
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_fp16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the FP16_WRAPPER can give the same result as when there is no hook registered.
        """
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        for hook, state in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.fp16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_bf16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the BF16_WRAPPER can give the same result as when there is no hook registered.
        """
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        for hook, state in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.bf16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether Python DDP communication hook POWER_SGD
        can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # Get GPU model with the hook registered.
        # Test the hook with different algorithmic configs.
        for use_error_feedback, warm_start, batch_tensors_with_same_shape in product(
            [True, False],
            [True, False],
            [True, False],
        ):
            state = powerSGD.PowerSGDState(
                process_group=process_group,
                matrix_approximation_rank=1,
                use_error_feedback=use_error_feedback,
                warm_start=warm_start,
                batch_tensors_with_same_shape=batch_tensors_with_same_shape,
            )
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                gpu_model = self._gpu_model_with_ddp_comm_hook(
                    process_group, hook, gradient_as_bucket_view, state
                )

                # check whether the grads are equal to what DDP without hook would return.
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_builtin_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether built-in C++ DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        for comm_hook_type in [
            dist.BuiltinCommHookType.ALLREDUCE,
            dist.BuiltinCommHookType.FP16_COMPRESS,
        ]:
            # Get GPU model with the built-in communication hook.
            gpu_model = self._gpu_model_with_builtin_ddp_comm_hook(
                process_group, comm_hook_type, gradient_as_bucket_view
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl(self):
        self._test_default_ddp_comm_hooks_nccl()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_nccl(self):
        self._test_fp16_compress_wrapper()

    @requires_accelerator_dist_backend()
    # @requires_nccl_version((2, 10), "Need NCCL 2.10+ for BF16_COMPRESS")
    @skip_but_pass_in_sandcastle_if(
        not BFLOAT16_AVAILABLE,
        "BFloat16 is only supported by CUDA 11+",
    )
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_nccl(self):
        self._test_bf16_compress_wrapper()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl(self):
        self._test_builtin_ddp_comm_hooks_nccl()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl(self):
        self._test_powerSGD_ddp_comm_hook_nccl()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_grad_is_view(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_static_graph(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(static_graph=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl_is_view(self):
        self._test_default_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_is_view(self):
        self._test_fp16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    # @requires_nccl_version((2, 10), "Need NCCL 2.10+ for BF16_COMPRESS")
    @skip_but_pass_in_sandcastle_if(
        not BFLOAT16_AVAILABLE,
        "BFloat16 is only supported by CUDA 11+",
    )
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_is_view(self):
        self._test_bf16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_nccl(self):
        """
        This unit test verifies whether a DDP communication hook that calls allreduce and then
        multiplies the result by ten and divides by two gives the expected result.
        """
        process_group = self._get_process_group()

        def allreduce_with_then_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # Multiply the result by 10.
                return 10 * fut.value()[0]

            def div(fut):
                # Divide the result by 2.
                return 0.5 * fut.value()

            return fut.then(mult).then(div)

        # Get GPU model with allreduce_with_then_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # check whether the grads are equal to what allreduce returns multiplied by 5.
        # without the comm_hook, result would be still 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))

    class AcceptsParam(torch.nn.Module):
        def __init__(self, p, factor):
            super().__init__()
            self.a = p
            self.f = factor

        def forward(self, input):
            return input + self.a * self.f

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_weight_sharing(self):
        process_group = self._get_process_group()

        size = 2048 * 2048
        dev = self.rank
        world = self.world_size

        p = torch.nn.Parameter(torch.randn(size, requires_grad=True))

        for try_set_to_none, use_bucket_view in product((False, True), (False, True)):
            m = torch.nn.Sequential(
                self.AcceptsParam(p, dev + 1), self.AcceptsParam(p, dev + 1)
            ).to(dev)

            m = torch.nn.parallel.DistributedDataParallel(
                m,
                bucket_cap_mb=1,
                gradient_as_bucket_view=use_bucket_view,
                device_ids=[dev],
                process_group=process_group,
            )

            for _ in range(3):
                m.zero_grad(set_to_none=try_set_to_none)
                m(1).sum().backward()

                # Each param value is multiplied by "rank + 1" twice in forward, so the grad
                # values produced by a particular rank should be 2. * (rank + 1).
                # Summing these over ranks and dividing by world size gives the expected result:
                analytic = torch.full_like(
                    p, 2.0 * (world * (world + 1.0) / 2.0) / world, device=dev
                )
                for name, p in m.named_parameters():
                    self.assertEqual(
                        p.grad,
                        analytic,
                        "mismatch at "
                        + name
                        + ".grad for "
                        + f"set_to_none = {try_set_to_none}, use_bucket_view = {use_bucket_view}",
                    )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_packed_sequence(self):
        """
        Tests that DDP with ``device_ids`` specified can run a forward and
        backward pass with ``PackedSequence`` s with parity compared to a local
        version of the model.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = dist.init_process_group(
            BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        seqs = ["sequence_sequence", "seq", "sequence"]
        vocab = ["<pad>"] + sorted({ch for seq in seqs for ch in seq})
        vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
        # Set the seed to make the embedding and LSTM deterministic (even
        # across ranks since DDP broadcasts parameters from rank 0)
        torch.manual_seed(0)
        embed = nn.Embedding(len(vocab), 4)  # keep on CPU
        lstm = nn.LSTM(input_size=4, hidden_size=2, batch_first=True).to(self.rank)
        lstm_ddp = DistributedDataParallel(
            copy.deepcopy(lstm),
            device_ids=[self.rank],
            process_group=process_group,
        )
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.module.parameters()):
            self.assertEqual(p1, p2)
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_tensor = torch.Tensor(
            torch.zeros((len(vectorized_seqs), seq_lengths.max()))
        ).long()
        for i, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[i, :seq_len] = torch.LongTensor(seq)
        seq_lengths, permutation_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[permutation_idx]
        embedded_seq_tensor = embed(seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor,
            seq_lengths,
            batch_first=True,
        )
        packed_input_ddp = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor.detach().clone(),
            seq_lengths,
            batch_first=True,
        )
        # Move the input to GPU explicitly for the local model
        packed_output, (ht, ct) = lstm(packed_input.to(self.rank))
        # Let DDP move the input to GPU internally
        packed_output_ddp, (ht_ddp, ct_ddp) = lstm_ddp(packed_input_ddp)
        self.assertEqual(packed_output.data, packed_output_ddp.data)
        self.assertEqual(ht, ht_ddp)
        self.assertEqual(ct, ct_ddp)
        packed_output.data.sum().backward()
        packed_output_ddp.data.sum().backward()
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_channels_last_contig(self):
        process_group = self._get_process_group()
        device = torch.device(f"{DEVICE_TYPE}:{self.rank}")
        tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(
            memory_format=torch.channels_last
        )
        process_group.broadcast([tensor]).wait()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_complex_params(self):
        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        N, C, H, W = 1, 16, 64, 64
        ddp_model = DistributedDataParallel(
            FFTModel(hin=H, win=W, n_features=C).to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

        inp = torch.ones((N, C, H, W), dtype=torch.float32)

        # train step
        out = ddp_model(inp)
        loss = torch.sum(out)
        loss.backward()
        optimizer.step()

        torch.accelerator.synchronize(device=device_id)


class WorkHookTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        if TEST_CUDA:
            # set TORCH_NCCL_ENABLE_TIMING to enable timing for CUDAEvents
            # in ProcessGroup Work
            os.environ["TORCH_NCCL_ENABLE_TIMING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        if TEST_CUDA:
            del os.environ["TORCH_NCCL_ENABLE_TIMING"]
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_broadcast(self):
        pg = self._get_process_group()
        num_hook_fired = 0
        durations: list[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).to(self.rank) * self.rank
        pg.broadcast([tensor]).wait()
        pg.broadcast([tensor]).wait()

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)

        self.assertEqual(num_hook_fired, 2)
        self.assertEqual(len(durations), 2)
        for duration in durations:
            self.assertTrue(duration > 0)

        self.assertEqual(tensor, torch.zeros([2, 3]).to(self.rank))

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_mixed_ops(self):
        pg = self._get_process_group()
        num_hook_fired = 0
        durations: list[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).to(self.rank)
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        # intentionally using async ops.
        pg.allreduce(tensor)
        pg.allgather(tensor_list, tensor)
        pg.allreduce(tensor)

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)

        self.assertEqual(num_hook_fired, 3)
        self.assertEqual(len(durations), 3)
        for duration in durations:
            self.assertTrue(duration > 0)

        self.assertEqual(
            tensor,
            torch.ones([2, 3]).to(self.rank) * self.world_size * self.world_size,
        )

        self.assertEqual(
            tensor_list,
            [
                torch.ones([2, 3]).to(self.rank) * self.world_size
                for _ in range(self.world_size)
            ],
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_with_ddp(self):
        pg = self._get_process_group()
        num_hook_fired: dict[int, int] = {}
        durations: dict[OpType, list[float]] = {}

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            num_hook_fired[op_type] += 1
            durations[op_type].append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)

        nlayers = 10
        net = nn.Sequential(
            *[nn.Linear(1000, 1000, bias=False) for _ in range(nlayers)]
        ).to(self.rank)

        ddp = DistributedDataParallel(
            net,
            device_ids=[self.rank],
            process_group=pg,
            bucket_cap_mb=1,
        )

        pg._wait_for_pending_works()

        # DDP is expected to synchronize model parameter by broadcasting
        # from rank0 to other ranks. However, this is DDP's internal implementation,
        # which is subject to change in future versions.
        self.assertTrue(num_hook_fired[OpType.BROADCAST] > 0)
        ctor_allreduce = (
            num_hook_fired[OpType.ALLREDUCE]
            if OpType.ALLREDUCE in num_hook_fired
            else 0
        )

        x = torch.zeros(2, 1000).to(self.rank)
        ddp(x).sum().backward()

        c10d.destroy_process_group(pg)

        self.assertTrue(OpType.ALLREDUCE in num_hook_fired)
        # The number of allreduce ops depend on DDP internal implementation, but
        # there should be at least one allreduce.
        self.assertTrue(num_hook_fired[OpType.ALLREDUCE] - ctor_allreduce > 0)
        self.assertTrue(all(duration > 0 for duration in chain(*(durations.values()))))

    # Not testing FSDP due to https://github.com/pytorch/pytorch/issues/90848.
    # We cannot disable workCleanupLoop() as hooks are fired in that thread.

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_all_gather_object(self):
        torch.accelerator.set_device_index(self.rank)

        pg = self._get_process_group()
        num_hook_fired: dict[int, int] = {}
        durations: dict[OpType, list[float]] = {}

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            num_hook_fired[op_type] += 1
            durations[op_type].append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)

        obj = {"rank": self.rank, "world_size": self.world_size}
        obj_list = [None for _ in range(self.world_size)]

        c10d.all_gather_object(obj_list, obj, group=pg)

        for r, o in enumerate(obj_list):
            self.assertTrue(isinstance(o, dict))
            self.assertTrue(set(o.keys()), {"rank", "world_size"})
            self.assertEqual(o["rank"], r)
            self.assertEqual(o["world_size"], self.world_size)

        c10d.destroy_process_group(pg)

        self.assertTrue(OpType.ALLGATHER in num_hook_fired)
        self.assertEqual(len(num_hook_fired), 1)
        # two allgathers, one for size and another for values
        self.assertEqual(num_hook_fired[OpType.ALLGATHER], 2)
        self.assertTrue(all(duration > 0 for duration in durations[OpType.ALLGATHER]))

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_seq(self):
        pg = self._get_process_group()
        num_hook_fired = 0
        seq: int = -1
        work: int = 0

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, seq
            num_hook_fired += 1
            seq = work_info.seq

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).to(self.rank) * self.rank
        work_count = 3
        for _ in range(work_count):
            work += 1
            pg.broadcast([tensor]).wait()

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)

        self.assertEqual(num_hook_fired, work_count)
        self.assertEqual(work, seq)


class NcclProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        self._test_collectives(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced(self):
        self._test_allreduce_coalesced(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(1)
    def test_all_to_all_single(self):
        self._test_all_to_all_single(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = DEVICE_TYPE
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(1)
    @parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_allgather_float8(self, float8_dtype):
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        if not sm_is_or_higher_than(device, 9, 0):  # noqa: F821
            self.skipTest("FP8 reduction support begins with sm90 capable devices")
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = DEVICE_TYPE
        tensor = torch.ones(10, 16, device=torch.device(device)).to(float8_dtype)
        output_tensor = torch.zeros(10, 16, device=torch.device(device)).to(
            float8_dtype
        )
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor.view(torch.float32), tensor.view(torch.float32))


instantiate_parametrized_tests(NcclProcessGroupWithDispatchedCollectivesTests)


class SetDeviceMethod(Enum):
    TORCH_CUDA_SET = auto()  # torch.accelerator.set_device_index
    COLLECTIVE_ARGUMENT = auto()  # broadcast_object_list(device=)


class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        if TEST_CUDA:
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
    def device(self):
        return self.rank

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync(self):
        self._test_new_group_local_sync(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_sanity_check(self):
        self._test_new_group_local_sync_sanity_check(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_duplicated_pg(self):
        self._test_new_group_local_sync_duplicate_pg(backend=BACKEND)

    def _init_two_pg2_subgroups(self, world_size: int = 4):
        if world_size != 4:
            raise NotImplementedError(
                f"need world size of 4 to get 2 subgroup PGs, but got world size of {world_size}"
            )
        store = c10d.FileStore(self.file_name, world_size)
        c10d.init_process_group(
            backend=BACKEND, store=store, rank=self.rank, world_size=world_size
        )
        # every rank creates the same sub groups
        # including unused sub groups in the current rank
        a_group = c10d.new_group([0, 1])
        b_group = c10d.new_group([2, 3])
        return a_group if self.rank < 2 else b_group

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_gather_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            # just easier to write the test for exactly 4 gpus, even if this test class increased to 8gpu later
            return

        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        input = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            gather_list = [torch.empty_like(input) for _ in range(subgroup.size())]
            if group_rank:
                # global_dst=0 group_dst=0 my_global_rank=2 gather_list is not None=True
                torch.distributed.gather(
                    input,
                    gather_list=gather_list,
                    group_dst=0,
                    group=subgroup,
                    async_op=False,
                )
            else:
                torch.distributed.gather(
                    input,
                    gather_list=gather_list,
                    dst=self.rank,
                    group=subgroup,
                    async_op=False,
                )
            for src in range(len(gather_list)):
                expected = (torch.ones_like(input) * self.rank) + src
                self.assertEqual(gather_list[src], expected)
        else:
            if group_rank:
                torch.distributed.gather(
                    input,
                    gather_list=None,
                    group_dst=0,
                    group=subgroup,
                    async_op=False,
                )
            else:
                torch.distributed.gather(
                    input,
                    gather_list=None,
                    dst=self.rank - 1,
                    group=subgroup,
                    async_op=False,
                )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_gather_object_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            # just easier to write the test for exactly 4 gpus, even if this test class increased to 8gpu later
            return

        subgroup = self._init_two_pg2_subgroups(world_size)

        # discrepancy #1
        # have to set device or else gather_object gets wrong device from 'current_device = _get_pg_default_device(group)
        torch.accelerator.set_device_index(self.rank)

        input = {"rank": self.rank}
        if self.rank == 0 or self.rank == 2:
            # discrepancy #2
            # another weird thing- what's the point of making me specify some empty objects in my list?
            # empty list should be valid imo.  (but it throws an error)
            gather_list = [{}, {}]
            if group_rank:
                torch.distributed.gather_object(
                    input, object_gather_list=gather_list, group_dst=0, group=subgroup
                )
            else:
                torch.distributed.gather_object(
                    input, object_gather_list=gather_list, dst=self.rank, group=subgroup
                )
            for src in range(len(gather_list)):
                self.assertEqual(gather_list[src]["rank"], self.rank + src)
        else:
            if group_rank:
                torch.distributed.gather_object(
                    input, object_gather_list=None, group_dst=0, group=subgroup
                )
            else:
                torch.distributed.gather_object(
                    input, object_gather_list=None, dst=self.rank - 1, group=subgroup
                )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_reduce_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        x = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            expected = x + torch.ones((10,), device=device) * (self.rank + 1)
            if group_rank:
                c10d.reduce(x, group_dst=0, group=subgroup, async_op=False)
            else:
                c10d.reduce(x, dst=self.rank, group=subgroup, async_op=False)
            self.assertEqual(x, expected)
        else:
            if group_rank:
                c10d.reduce(x, group_dst=0, group=subgroup, async_op=False)
            else:
                c10d.reduce(x, dst=self.rank - 1, group=subgroup, async_op=False)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    @parametrize("async_op", [True, False])
    def test_send_recv_subgroup(self, async_op, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            if async_op:
                if group_rank:
                    c10d.irecv(x, group_src=1, group=subgroup).wait()
                else:
                    c10d.irecv(x, src=self.rank + 1, group=subgroup).wait()
            else:
                if group_rank:
                    c10d.recv(x, group_src=1, group=subgroup)
                else:
                    c10d.recv(x, src=self.rank + 1, group=subgroup)
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            if async_op:
                if group_rank:
                    c10d.isend(x, group_dst=0, group=subgroup).wait()
                else:
                    c10d.isend(x, dst=self.rank - 1, group=subgroup).wait()
            else:
                if group_rank:
                    c10d.send(x, group_dst=0, group=subgroup)
                else:
                    c10d.send(x, dst=self.rank - 1, group=subgroup)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_batch_send_recv_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        ops = []
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            if group_rank:
                ops.append(c10d.P2POp(dist.irecv, x, group=subgroup, group_peer=1))
            else:
                ops.append(
                    c10d.P2POp(dist.irecv, x, peer=self.rank + 1, group=subgroup)
                )

            for work in dist.batch_isend_irecv(ops):
                work.wait()
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            if group_rank:
                ops.append(c10d.P2POp(dist.isend, x, group=subgroup, group_peer=0))
            else:
                ops.append(
                    c10d.P2POp(dist.isend, x, peer=self.rank - 1, group=subgroup)
                )
            for work in dist.batch_isend_irecv(ops):
                work.wait()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_broadcast_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            if group_rank:
                c10d.broadcast(x, group_src=1, group=subgroup)
            else:
                c10d.broadcast(x, src=self.rank + 1, group=subgroup)
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            if group_rank:
                c10d.broadcast(x, group_src=1, group=subgroup)
            else:
                c10d.broadcast(x, src=self.rank, group=subgroup)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device_index",
        [SetDeviceMethod.TORCH_CUDA_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    @parametrize("group_rank", [True, False])
    def test_send_recv_object_list_subgroup(
        self, set_device_index: SetDeviceMethod, group_rank
    ):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        if set_device_index == SetDeviceMethod.TORCH_CUDA_SET:
            torch.accelerator.set_device_index(self.rank)
            device = None
        else:
            device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = [{}]
            if group_rank:
                c10d.recv_object_list(x, group_src=1, group=subgroup, device=device)
            else:
                c10d.recv_object_list(
                    x, src=self.rank + 1, group=subgroup, device=device
                )
            expected = [{"rank": self.rank + 1}]
            self.assertEqual(x, expected)
        else:
            x = [{"rank": self.rank}]
            if group_rank:
                c10d.send_object_list(x, group_dst=0, group=subgroup, device=device)
            else:
                c10d.send_object_list(
                    x, dst=self.rank - 1, group=subgroup, device=device
                )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device_index",
        [SetDeviceMethod.TORCH_CUDA_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    @parametrize("group_rank", [True, False])
    def test_broadcast_object_list_subgroup(
        self, set_device_index: SetDeviceMethod, group_rank
    ):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        if set_device_index == SetDeviceMethod.TORCH_CUDA_SET:
            torch.accelerator.set_device_index(self.rank)
            device = None
        else:
            device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = [{}]
            if group_rank:
                c10d.broadcast_object_list(
                    x, group_src=1, group=subgroup, device=device
                )
            else:
                c10d.broadcast_object_list(
                    x, src=self.rank + 1, group=subgroup, device=device
                )
            expected = [{"rank": self.rank + 1}]
            self.assertEqual(x, expected)
        else:
            x = [{"rank": self.rank}]
            if group_rank:
                c10d.broadcast_object_list(
                    x, group_src=1, group=subgroup, device=device
                )
            else:
                c10d.broadcast_object_list(
                    x, src=self.rank, group=subgroup, device=device
                )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_scatter_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        x = torch.empty((10,), device=device)
        expected = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            if group_rank:
                c10d.scatter(x, scatter_list=None, group_src=1, group=subgroup)
            else:
                c10d.scatter(x, scatter_list=None, src=self.rank + 1, group=subgroup)
        else:
            scatter_list = [
                torch.ones((10,), device=device) * (self.rank - 1),
                torch.ones((10,), device=device) * self.rank,
            ]
            if group_rank:
                c10d.scatter(x, scatter_list=scatter_list, group_src=1, group=subgroup)
            else:
                c10d.scatter(
                    x, scatter_list=scatter_list, src=self.rank, group=subgroup
                )
        self.assertEqual(x, expected)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_scatter_object_list_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        torch.accelerator.set_device_index(self.rank)
        scatter_object_output_list = [None]
        expected = [{"rank": self.rank}]
        if self.rank == 0 or self.rank == 2:
            if group_rank:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=None,
                    group_src=1,
                    group=subgroup,
                )
            else:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=None,
                    src=self.rank + 1,
                    group=subgroup,
                )

        else:
            scatter_object_input_list = [
                {"rank": self.rank - 1},
                {"rank": self.rank},
            ]
            if group_rank:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=scatter_object_input_list,
                    group_src=1,
                    group=subgroup,
                )
            else:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=scatter_object_input_list,
                    src=self.rank,
                    group=subgroup,
                )
        self.assertEqual(scatter_object_output_list, expected)


instantiate_parametrized_tests(LargeCommTest)


class SparseCollective(MultiProcessTestCase):
    @property
    def world_size(self):
        return 1

    def setUp(self):
        super().setUp()
        if TEST_CUDA:
            # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
            # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.accelerator.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    class ToyModel(nn.Module):
        def __init__(self, rank, vocab_size, embedding_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True).to(
                rank
            )
            self.linear = nn.Linear(embedding_dim, 1).to(rank)

        def forward(self, inputs):
            embedded = self.embedding(inputs)
            # embedded shape: (batch_size, sequence_length, embedding_dim)
            flattened = torch.mean(embedded, dim=1)
            # flattened shape: (batch_size, embedding_dim)
            output = self.linear(flattened)
            # output shape: (batch_size, 1)
            return output

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(1)
    def test_ddp_set_sparse_metadata(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        vocab_size = 5

        model = SparseCollective.ToyModel(
            self.rank, vocab_size=vocab_size, embedding_dim=10
        )
        ddp_model = DistributedDataParallel(model)
        inputs = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]).to(self.rank)
        # set sparse metadata on the DDP model
        indices = torch.Tensor(list(range(vocab_size)))
        ddp_model._set_sparse_metadata({"embedding.weight": indices})
        # forward pass
        try:
            output = ddp_model(inputs)
            loss = output.sum()

            # backward pass
            loss.backward()
            self.assertTrue(ddp_model.module.embedding.weight.grad.indices, indices)
        except RuntimeError as e:
            if "NCCL does not support all_reduce with sparse tensors" in str(e):
                pass
            else:
                # Rethrow the exception if it's a different error
                raise

