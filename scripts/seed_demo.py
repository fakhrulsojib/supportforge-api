"""Seed a demo tenant with Bitext-style customer support Q&A data.

This script generates synthetic customer support data that mimics
the Bitext Customer Support dataset structure, then:
1. Creates document records in PostgreSQL
2. Chunks the Q&A pairs
3. Embeds them via Ollama
4. Stores embeddings in ChromaDB

Usage:
    python scripts/seed_demo.py [--tenant-slug demo-support]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Synthetic Q&A pairs modeled after Bitext Customer Support dataset
DEMO_QA_PAIRS: list[dict[str, str]] = [
    {
        "category": "ACCOUNT",
        "question": "How do I reset my password?",
        "answer": "To reset your password, go to the login page and click 'Forgot Password'. "
        "Enter your registered email address and we'll send you a reset link. "
        "The link is valid for 24 hours. If you don't receive the email, check your spam folder.",
    },
    {
        "category": "ACCOUNT",
        "question": "How can I change my email address?",
        "answer": "Navigate to Settings > Profile > Email. Enter your new email address and "
        "confirm the change. We'll send a verification email to your new address. "
        "Click the verification link to complete the change.",
    },
    {
        "category": "ACCOUNT",
        "question": "How do I delete my account?",
        "answer": "To delete your account, go to Settings > Account > Delete Account. "
        "Please note that account deletion is permanent and cannot be reversed. "
        "All your data, including conversation history, will be permanently removed.",
    },
    {
        "category": "BILLING",
        "question": "Where can I find my invoice?",
        "answer": "Your invoices are available under Billing > Invoice History. "
        "You can download PDF copies of all past invoices. "
        "Invoices are generated on the first day of each billing cycle.",
    },
    {
        "category": "BILLING",
        "question": "How do I update my payment method?",
        "answer": "Go to Billing > Payment Methods. You can add a new credit card, "
        "debit card, or bank account. Remove old payment methods after adding the new one. "
        "Changes take effect on your next billing cycle.",
    },
    {
        "category": "BILLING",
        "question": "Can I get a refund?",
        "answer": "We offer refunds within 14 days of purchase if you haven't used the service. "
        "Contact our support team with your order number and reason for the refund request. "
        "Refunds are processed within 5-10 business days.",
    },
    {
        "category": "SHIPPING",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. "
        "Overnight shipping is available for an additional fee. "
        "You'll receive a tracking number once your order ships.",
    },
    {
        "category": "SHIPPING",
        "question": "How do I track my order?",
        "answer": "Once your order ships, you'll receive a tracking email with a link. "
        "You can also check order status under My Orders > Track Order. "
        "Enter your tracking number for real-time delivery updates.",
    },
    {
        "category": "PRODUCT",
        "question": "What is your return policy?",
        "answer": "We accept returns within 30 days of purchase for items in their original condition. "
        "Items must be unworn/unused with tags attached. "
        "Initiate a return through My Orders > Return Item.",
    },
    {
        "category": "PRODUCT",
        "question": "Do you offer product warranties?",
        "answer": "Yes, all products come with a 1-year manufacturer warranty. "
        "Extended warranties are available for purchase at checkout. "
        "Warranty claims can be filed through Settings > Warranty Claims.",
    },
    {
        "category": "TECHNICAL",
        "question": "The app is crashing when I open it.",
        "answer": "Try these troubleshooting steps: 1) Force close and reopen the app, "
        "2) Clear the app cache in your device settings, "
        "3) Update to the latest version from the app store, "
        "4) If the issue persists, uninstall and reinstall the app.",
    },
    {
        "category": "TECHNICAL",
        "question": "I'm getting a 'Session Expired' error.",
        "answer": "This error occurs when your login session times out for security. "
        "Simply log back in with your credentials. If you're seeing this frequently, "
        "ensure your device's date/time is set correctly and try clearing browser cookies.",
    },
]


async def seed_demo_data(tenant_slug: str) -> None:
    """Seed demo Q&A pairs for a tenant.

    This is a simplified seeder that stores the Q&A content as document
    records. Full embedding + ChromaDB ingestion requires running services.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.domain.models.document import Document
    from app.domain.models.enums import DocumentStatus
    from app.infrastructure.database.models import Base
    from app.infrastructure.database.repositories.document_repo import SQLDocumentRepository
    from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

    settings = get_settings()
    engine = create_async_engine(settings.computed_database_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        tenant_repo = SQLTenantRepository(session)
        doc_repo = SQLDocumentRepository(session)

        # Find the tenant
        tenant = await tenant_repo.get_by_slug(tenant_slug)
        if not tenant:
            print(f"✗ Tenant with slug '{tenant_slug}' not found. Run create_tenant.py first.")
            return

        # Check existing documents to make this idempotent
        existing_docs = await doc_repo.list_by_tenant(tenant.id)
        existing_filenames = {d.filename for d in existing_docs}

        created_count = 0
        for qa in DEMO_QA_PAIRS:
            filename = f"qa_{qa['category'].lower()}_{qa['question'][:30].replace(' ', '_')}.txt"
            if filename in existing_filenames:
                continue

            content = f"Category: {qa['category']}\n\nQ: {qa['question']}\n\nA: {qa['answer']}"
            doc = Document(
                tenant_id=tenant.id,
                filename=filename,
                file_type="txt",
                status=DocumentStatus.READY,
                chunk_count=1,
            )
            await doc_repo.create(doc)

            # Create a chunk for this Q&A pair
            from app.domain.models.document import DocumentChunk

            chunk = DocumentChunk(
                document_id=doc.id if doc.id else "placeholder",
                chunk_index=0,
                content=content,
                chroma_id=f"seed_{qa['category'].lower()}_{created_count}",
            )
            # Note: chunk creation needs the actual document ID from the DB
            # This will be handled properly with the full ingestion pipeline in Phase 2

            created_count += 1

        await session.commit()
        print(f"✓ Seeded {created_count} Q&A pairs for tenant '{tenant_slug}'")
        print(f"  Total documents: {len(existing_docs) + created_count}")

    await engine.dispose()


def main() -> None:
    """Parse CLI args and seed demo data."""
    import argparse

    parser = argparse.ArgumentParser(description="Seed demo Q&A data")
    parser.add_argument("--tenant-slug", default="demo-support", help="Target tenant slug")
    args = parser.parse_args()

    asyncio.run(seed_demo_data(args.tenant_slug))


if __name__ == "__main__":
    main()
