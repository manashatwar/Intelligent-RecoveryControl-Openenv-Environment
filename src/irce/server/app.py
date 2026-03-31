from __future__ import annotations

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:  # pragma: no cover
    from openenv_core.env_server import create_fastapi_app

try:
    from irce.environment import IRCEEnv
    from irce.models import IRCEAction, IRCEObservation
except ImportError:  # pragma: no cover
    from environment import IRCEEnv
    from models import IRCEAction, IRCEObservation


def create_environment() -> IRCEEnv:
    return IRCEEnv()


app = create_fastapi_app(create_environment, IRCEAction, IRCEObservation)
