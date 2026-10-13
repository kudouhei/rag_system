"""Local development identities for training management."""
from __future__ import annotations

from secrets import compare_digest
from typing import Annotated, Literal

from fastapi import Depends, HTTPException
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from pydantic import BaseModel, ConfigDict, Field

from app.core import config


class TrainingActor(BaseModel):
    """Tenant and role established by backend authentication."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tenant_id: str = Field(
        min_length=1,
        max_length=100,
    )
    actor_id: str = Field(
        min_length=1,
        max_length=200,
    )
    actor_role: Literal[
        "reviewer",
        "publisher",
        "administrator",
    ]


_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="TrainingDevelopmentBearer",
)


def get_training_actor(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer),
    ],
) -> TrainingActor:
    if (
        config.APP_ENV != "development"
        or not config.TRAINING_DEV_AUTH_ENABLED
    ):
        raise HTTPException(
            status_code=404,
            detail="Not Found",
        )

    profiles: list[
        tuple[
            Literal[
                "reviewer",
                "publisher",
                "administrator",
            ],
            str,
        ]
    ] = [
        (
            "reviewer",
            config.TRAINING_DEV_REVIEWER_TOKEN,
        ),
        (
            "publisher",
            config.TRAINING_DEV_PUBLISHER_TOKEN,
        ),
        (
            "administrator",
            config.TRAINING_DEV_ADMIN_TOKEN,
        ),
    ]

    configured_profiles = [
        (role, token)
        for role, token in profiles
        if token
    ]

    tokens = [
        token
        for _, token in configured_profiles
    ]

    if (
        not config.TRAINING_DEV_TENANT_ID
        or not configured_profiles
        or any(len(token) < 32 for token in tokens)
        or len(tokens) != len(set(tokens))
    ):
        raise HTTPException(
            status_code=503,
            detail="Training development authentication "
            "is not configured correctly.",
        )

    if credentials is not None:
        supplied_token = credentials.credentials.encode(
            "utf-8"
        )

        for role, configured_token in configured_profiles:
            if compare_digest(
                supplied_token,
                configured_token.encode("utf-8"),
            ):
                return TrainingActor(
                    tenant_id=config.TRAINING_DEV_TENANT_ID,
                    actor_id=f"local-{role}",
                    actor_role=role,
                )

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid training credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )