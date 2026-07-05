from __future__ import annotations

from typing import Any

import pymysql

from FinanceAgent.config import settings


def connect_runtime_mysql(*, pymysql_module: Any | None = None):
    module = pymysql_module or pymysql
    try:
        return _connect(module, database=settings.mysql_database)
    except module.err.OperationalError as exc:
        if _mysql_error_code(exc) != 1049:
            raise
        ensure_runtime_database(pymysql_module=module)
        return _connect(module, database=settings.mysql_database)


def ensure_runtime_database(*, pymysql_module: Any | None = None) -> None:
    module = pymysql_module or pymysql
    with _connect(module, database=None) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"create database if not exists {_quote_identifier(settings.mysql_database)} "
                "character set utf8mb4 collate utf8mb4_unicode_ci"
            )


def _connect(module: Any, *, database: str | None):
    return module.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=database,
        charset="utf8mb4",
        autocommit=True,
    )


def _mysql_error_code(exc: BaseException) -> int | None:
    if not getattr(exc, "args", None):
        return None
    code = exc.args[0]
    return int(code) if isinstance(code, int) else None


def _quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"
