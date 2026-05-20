from types import SimpleNamespace

import pytest

from slime.ray import rollout


class FakeRemoteMethod:
    def __init__(self, node_ip):
        self.node_ip = node_ip
        self.calls = []

    def remote(self, *, start_port, consecutives):
        self.calls.append({"start_port": start_port, "consecutives": list(consecutives)})
        ports = []
        next_port = start_port
        for consecutive in consecutives:
            ports.append(next_port)
            next_port += consecutive
        return self.node_ip, ports, next_port


class FakeEngine:
    def __init__(self, node_ip):
        self._get_current_node_ip_and_free_ports = FakeRemoteMethod(node_ip)


@pytest.fixture(autouse=True)
def fake_ray_get(monkeypatch):
    monkeypatch.setattr(rollout.ray, "get", lambda value: value)


def make_args(**overrides):
    values = {
        "num_gpus_per_node": 4,
        "rollout_num_gpus_per_engine": 2,
        "sglang_dp_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_batched_port_allocation_for_regular_engines_on_one_node():
    args = make_args()
    engine0 = FakeEngine("10.0.0.1")
    engine1 = FakeEngine("10.0.0.1")

    addr_and_ports, cursors = rollout._allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=[(0, engine0), (1, engine1)],
        base_port=15000,
    )

    assert engine0._get_current_node_ip_and_free_ports.calls == [
        {"start_port": 15000, "consecutives": [1, 1, 1, 1, 31, 31]}
    ]
    assert engine1._get_current_node_ip_and_free_ports.calls == []
    assert addr_and_ports == {
        0: {"host": "10.0.0.1", "port": 15000, "nccl_port": 15001, "dist_init_addr": "10.0.0.1:15004"},
        1: {"host": "10.0.0.1", "port": 15002, "nccl_port": 15003, "dist_init_addr": "10.0.0.1:15035"},
    }
    assert cursors == {0: 15066}


def test_batched_port_allocation_includes_prefill_bootstrap_ports():
    args = make_args(num_gpus_per_node=2, rollout_num_gpus_per_engine=2)
    engine = FakeEngine("10.0.0.2")

    addr_and_ports, cursors = rollout._allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=[(0, engine)],
        worker_type="prefill",
        base_port=16000,
    )

    assert engine._get_current_node_ip_and_free_ports.calls == [
        {"start_port": 16000, "consecutives": [1, 1, 1, 31]}
    ]
    assert addr_and_ports == {
        0: {
            "host": "10.0.0.2",
            "port": 16000,
            "nccl_port": 16001,
            "disaggregation_bootstrap_port": 16002,
            "dist_init_addr": "10.0.0.2:16003",
        }
    }
    assert cursors == {0: 16034}


def test_batched_port_allocation_preserves_multi_node_dist_init_addr():
    args = make_args(num_gpus_per_node=1, rollout_num_gpus_per_engine=2)
    engine0 = FakeEngine("10.0.0.3")
    engine1 = FakeEngine("10.0.0.4")

    addr_and_ports, cursors = rollout._allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=[(0, engine0), (1, engine1)],
        base_port=17000,
    )

    assert engine0._get_current_node_ip_and_free_ports.calls == [
        {"start_port": 17000, "consecutives": [1, 1, 31]}
    ]
    assert engine1._get_current_node_ip_and_free_ports.calls == [
        {"start_port": 17000, "consecutives": [1, 1]}
    ]
    assert addr_and_ports == {
        0: {"host": "10.0.0.3", "port": 17000, "nccl_port": 17001, "dist_init_addr": "10.0.0.3:17002"},
        1: {"host": "10.0.0.4", "port": 17000, "nccl_port": 17001, "dist_init_addr": "10.0.0.3:17002"},
    }
    assert cursors == {0: 17033, 1: 17002}


def test_external_rollout_wrappers_do_not_reserve_gpu_bundles(monkeypatch):
    created_actors = []
    option_calls = []

    class FakeInitMethod:
        def __init__(self):
            self.calls = []

        def remote(self, **kwargs):
            self.calls.append(kwargs)
            return ("init", kwargs)

    class FakeActor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.init = FakeInitMethod()

    class FakeRemoteActor:
        def options(self, **kwargs):
            option_calls.append(kwargs)
            return self

        def remote(self, *args, **kwargs):
            actor = FakeActor(*args, **kwargs)
            created_actors.append(actor)
            return actor

    monkeypatch.setattr(rollout.ray, "remote", lambda _cls: FakeRemoteActor())

    args = make_args(
        debug_train_only=False,
        rollout_external=True,
        rollout_external_engine_addrs=["10.0.0.5:30000"],
    )
    group = rollout.ServerGroup(
        args=args,
        pg=(object(), [0], [3]),
        all_engines=[None],
        num_gpus_per_engine=2,
        num_new_engines=0,
        worker_type="regular",
    )

    handles, cursors = group.start_engines()

    assert cursors == {}
    assert handles == [
        (
            "init",
            {
                "dist_init_addr": "10.0.0.5:30000",
                "nccl_port": None,
                "host": "10.0.0.5",
                "port": 30000,
                "router_ip": None,
                "router_port": None,
            },
        )
    ]
    assert len(created_actors) == 1
    assert option_calls[0]["num_gpus"] == 0
    assert option_calls[0]["num_cpus"] == 0.2
    assert "scheduling_strategy" not in option_calls[0]
    assert created_actors[0].kwargs["base_gpu_id"] == 3
