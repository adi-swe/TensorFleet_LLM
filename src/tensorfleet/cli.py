"""CLI distinguishes measured execution, simulated scenarios and held-out validation."""

import argparse
import json
import shlex


def main():
    parser = argparse.ArgumentParser(prog="tensorfleet")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output", required=True)
    for name, default in (
        ("steps", 10),
        ("batch-size", 4),
        ("checkpoint-every", 2),
        ("max-restarts", 1),
        ("world-size", None),
        ("dataset-size", 64),
        ("straggler-rank", None),
        ("straggler-start", 0),
    ):
        train.add_argument(f"--{name}", type=int, default=default)
    train.add_argument("--fail-step", type=int)
    train.add_argument("--fail-rank", type=int)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--stochastic-data", action="store_true")
    train.add_argument("--straggler-seconds", type=float, default=0.0)
    train.add_argument("--straggler-stop", type=int)
    train.add_argument("--agent-command", help="trusted executable adapter, parsed without a shell")
    train.add_argument("--agent-timeout", type=float, default=5.0)
    simulation = commands.add_parser("simulate")
    simulation.add_argument("--profile", required=True)
    simulation.add_argument("--steps", type=int, default=10)
    simulation.add_argument("--checkpoint-every", type=int, default=2)
    simulation.add_argument("--fail-step", type=int)
    simulation.add_argument("--restart-seconds", type=float)
    simulation.add_argument("--checkpoint-seconds", type=float)
    simulation.add_argument("--contention-factor", type=float, default=1.0)
    simulation.add_argument(
        "--cost-profile", action="store_true", help="include profiled lifecycle and I/O costs"
    )
    capacity = commands.add_parser("simulate-contention")
    capacity.add_argument("--profile", required=True)
    capacity.add_argument("--job-steps", type=int, nargs="+", required=True)
    capacity.add_argument("--world-size", type=int, default=2)
    capacity.add_argument("--slots", type=int, default=2)
    validate = commands.add_parser("validate-simulation")
    validate.add_argument("--calibration", required=True)
    validate.add_argument("--holdout", nargs="+", required=True)
    validate.add_argument("--tolerance", type=float, default=0.25)
    campaign = commands.add_parser("benchmark-simulation")
    campaign.add_argument("--output", required=True)
    campaign.add_argument("--steps", type=int, default=30)
    campaign.add_argument("--repeats", type=int, default=3)
    campaign.add_argument("--delay-seconds", type=float, default=0.005)
    campaign.add_argument("--tolerance", type=float, default=0.25)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        from tensorfleet.executor.device import resolve_config
        from tensorfleet.orchestration.agents import CommandAgentRuntime, GuardedAgentController
        from tensorfleet.orchestration.supervisor import run_training

        output = args.pop("output")
        agent_command = args.pop("agent_command")
        timeout = args.pop("agent_timeout")
        controller = (
            GuardedAgentController(CommandAgentRuntime(shlex.split(agent_command), timeout))
            if agent_command
            else None
        )
        result = run_training(resolve_config(**args), output, controller)
    elif command == "simulate":
        from tensorfleet.simulator.engine import load_profile, simulate
        from tensorfleet.simulator.profile import load_timing_profile
        from tensorfleet.simulator.validation import predict

        cost_profile = args.pop("cost_profile")
        directory = args.pop("profile")
        if cost_profile:
            profile = load_timing_profile(directory)
            config = dict(profile.config)
            for name in ("steps", "checkpoint_every", "fail_step"):
                config[name] = args[name]
            result = predict(
                profile,
                config,
                contention_factor=args["contention_factor"],
                restart_seconds=args["restart_seconds"],
                checkpoint_seconds=args["checkpoint_seconds"],
            ).to_dict()
        else:
            args["restart_seconds"] = (
                1.0 if args["restart_seconds"] is None else args["restart_seconds"]
            )
            args["checkpoint_seconds"] = (
                0.0 if args["checkpoint_seconds"] is None else args["checkpoint_seconds"]
            )
            result = simulate(load_profile(directory), **args).to_dict()
    elif command == "simulate-contention":
        from tensorfleet.simulator.contention import simulate_shared_capacity
        from tensorfleet.simulator.engine import load_profile

        result = simulate_shared_capacity(
            load_profile(args["profile"]), args["job_steps"], args["world_size"], args["slots"]
        )
    elif command == "validate-simulation":
        from tensorfleet.simulator.validation import validate_profile

        result = validate_profile(args["calibration"], args["holdout"], args["tolerance"])
    else:
        from tensorfleet.simulator.validation import run_validation_campaign

        result = run_validation_campaign(**args)
    print(json.dumps(result, indent=2))
    if command in ("validate-simulation", "benchmark-simulation") and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
