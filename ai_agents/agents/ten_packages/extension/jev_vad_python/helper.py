
import json

from typing import Any

from ten_runtime import AsyncTenEnv, Data, Cmd


async def _send_data(ten_env: AsyncTenEnv, data_name: str, payload: Any | None) -> None:
    data = Data.create(
        name=data_name,
    )

    if payload is not None:
        data.set_property_from_json(None, json.dumps(payload))
    ten_env.log_debug(f"send_data: data_name {data_name} and payload: {json.dumps(payload)}")
    await ten_env.send_data(data)

async def _send_cmd(ten_env: AsyncTenEnv, cmd_name: str, payload: Any | None) -> None:
    cmd = Cmd.create(
        name=cmd_name,
    )

    if payload is not None:
        cmd.set_property_from_json(None, json.dumps(payload))
    ten_env.log_debug(f"send_cmd: cmd_name {cmd_name} and payload: {json.dumps(payload)}")
    await ten_env.send_cmd(cmd)