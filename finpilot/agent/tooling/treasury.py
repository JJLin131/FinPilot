from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from finpilot.models import GraphState


class AccountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accountId: StrictStr = Field(min_length=1, max_length=64)


class TransactionsArgs(AccountArgs):
    startDate: date | None = None
    endDate: date | None = None
    limit: StrictInt = Field(default=10, ge=1, le=100)


class ReceiptStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactionId: StrictStr = Field(min_length=1, max_length=64)


class CashPoolPositionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poolId: StrictStr | None = Field(default=None, min_length=1, max_length=64)


class PaymentOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accountId: StrictStr = Field(min_length=1, max_length=64)
    payeeAccountId: StrictStr = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=1_000_000_000)
    currency: StrictStr = Field(default="CNY", min_length=3, max_length=3)
    purpose: StrictStr = Field(min_length=1, max_length=256)


class TransferOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fromAccountId: StrictStr = Field(min_length=1, max_length=64)
    toAccountId: StrictStr = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=1_000_000_000)
    currency: StrictStr = Field(default="CNY", min_length=3, max_length=3)
    purpose: StrictStr = Field(min_length=1, max_length=256)


class DownloadReceiptArgs(ReceiptStatusArgs):
    pass


def register_treasury_tools(registry) -> None:
    from finpilot.agent.tools import ToolSpec

    registry.register(
        ToolSpec(
            name="query_treasury_account",
            description="Query treasury account profile and status by accountId.",
            when_to_use="Use for read-only account profile, bank, currency, status, and ownership queries.",
            arguments={"accountId": "string"},
            args_model=AccountArgs,
            executor=_query_treasury_account,
            risk_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="query_account_balance",
            description="Query treasury account balance by accountId.",
            when_to_use="Use for read-only current balance, available balance, frozen balance, and currency queries.",
            arguments={"accountId": "string"},
            args_model=AccountArgs,
            executor=_query_account_balance,
            risk_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="query_transactions",
            description="Query treasury transaction records by accountId and optional date range.",
            when_to_use="Use for read-only bank statement, transaction flow, income, expense, or reconciliation queries.",
            arguments={"accountId": "string", "startDate": "date, optional", "endDate": "date, optional", "limit": "integer, optional"},
            args_model=TransactionsArgs,
            executor=_query_transactions,
            risk_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="query_receipt_status",
            description="Query receipt status for a transaction by transactionId.",
            when_to_use="Use for read-only receipt generation, availability, or download status queries.",
            arguments={"transactionId": "string"},
            args_model=ReceiptStatusArgs,
            executor=_query_receipt_status,
            risk_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="query_cash_pool_position",
            description="Query cash pool position and member account summary.",
            when_to_use="Use for read-only cash pool balance, collection, allocation, or member position queries.",
            arguments={"poolId": "string, optional"},
            args_model=CashPoolPositionArgs,
            executor=_query_cash_pool_position,
            risk_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="create_payment_order",
            description="Create a mock payment order request; no real money is moved.",
            when_to_use="Use only when the user asks to create, draft, or initiate a payment order.",
            arguments={
                "accountId": "string",
                "payeeAccountId": "string",
                "amount": "number",
                "currency": "string, optional",
                "purpose": "string",
            },
            args_model=PaymentOrderArgs,
            executor=_create_payment_order,
            risk_level="high",
        )
    )
    registry.register(
        ToolSpec(
            name="create_transfer_order",
            description="Create a mock transfer order request; no real money is moved.",
            when_to_use="Use only when the user asks to create, draft, or initiate a funds transfer.",
            arguments={
                "fromAccountId": "string",
                "toAccountId": "string",
                "amount": "number",
                "currency": "string, optional",
                "purpose": "string",
            },
            args_model=TransferOrderArgs,
            executor=_create_transfer_order,
            risk_level="high",
        )
    )
    registry.register(
        ToolSpec(
            name="download_receipt",
            description="Create a mock receipt download request by transactionId.",
            when_to_use="Use when the user asks to download a transaction receipt or voucher document.",
            arguments={"transactionId": "string"},
            args_model=DownloadReceiptArgs,
            executor=_download_receipt,
            risk_level="medium",
        )
    )


def _content(tool_name: str, title: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "content": [
            {
                "source": "mock://treasury-backend",
                "title": title,
                "text": text,
                "metadata": {"kind": "treasury_data", **metadata},
            }
        ],
    }


def _query_treasury_account(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    account_id = parameters["accountId"]
    return _content(
        "query_treasury_account",
        f"Account {account_id}",
        f"Mock account {account_id}: bank=ICBC, accountName=Operating Account, status=ACTIVE, currency=CNY.",
        {"accountId": account_id, "status": "ACTIVE", "currency": "CNY"},
    )


def _query_account_balance(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    account_id = parameters["accountId"]
    return _content(
        "query_account_balance",
        f"Balance {account_id}",
        f"Mock balance for {account_id}: currentBalance=1250000.00, availableBalance=1180000.00, frozenBalance=70000.00, currency=CNY.",
        {
            "accountId": account_id,
            "currentBalance": 1_250_000.00,
            "availableBalance": 1_180_000.00,
            "frozenBalance": 70_000.00,
            "currency": "CNY",
        },
    )


def _query_transactions(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    account_id = parameters["accountId"]
    return _content(
        "query_transactions",
        f"Transactions {account_id}",
        f"Mock transactions for {account_id}: TXN-1001 income 500000.00 CNY; TXN-1002 expense 120000.00 CNY.",
        {
            "accountId": account_id,
            "transactions": [
                {"transactionId": "TXN-1001", "direction": "IN", "amount": 500_000.00, "currency": "CNY"},
                {"transactionId": "TXN-1002", "direction": "OUT", "amount": 120_000.00, "currency": "CNY"},
            ],
        },
    )


def _query_receipt_status(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    transaction_id = parameters["transactionId"]
    return _content(
        "query_receipt_status",
        f"Receipt {transaction_id}",
        f"Mock receipt status for {transaction_id}: status=AVAILABLE, downloadable=true.",
        {"transactionId": transaction_id, "status": "AVAILABLE", "downloadable": True},
    )


def _query_cash_pool_position(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    pool_id = parameters.get("poolId") or "POOL-DEFAULT"
    return _content(
        "query_cash_pool_position",
        f"Cash Pool {pool_id}",
        f"Mock cash pool {pool_id}: totalBalance=5800000.00 CNY, collectedToday=900000.00 CNY, allocatedToday=350000.00 CNY.",
        {
            "poolId": pool_id,
            "totalBalance": 5_800_000.00,
            "collectedToday": 900_000.00,
            "allocatedToday": 350_000.00,
            "currency": "CNY",
        },
    )


def _create_payment_order(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "mock": True,
        "operation": "create_payment_order",
        "orderId": "PAY-MOCK-001",
        "status": "DRAFT_CREATED",
        "message": "Mock payment order created; no money was moved.",
        **parameters,
    }


def _create_transfer_order(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "mock": True,
        "operation": "create_transfer_order",
        "orderId": "TRF-MOCK-001",
        "status": "DRAFT_CREATED",
        "message": "Mock transfer order created; no money was moved.",
        **parameters,
    }


def _download_receipt(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    transaction_id = parameters["transactionId"]
    return {
        "mock": True,
        "operation": "download_receipt",
        "transactionId": transaction_id,
        "status": "READY",
        "downloadUrl": f"mock://receipts/{transaction_id}.pdf",
        "message": "Mock receipt download prepared.",
    }
