"""Add model_type default value

Revision ID: 002
Revises: 001
Create Date: 2026-02-19 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Set default value for existing rows
    op.execute("ALTER TABLE styles ALTER COLUMN model_type SET DEFAULT 'nilut'")


def downgrade() -> None:
    # Remove default value
    op.execute("ALTER TABLE styles ALTER COLUMN model_type DROP DEFAULT")
