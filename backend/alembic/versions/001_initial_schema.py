"""Initial database schema

Revision ID: 001
Create Date: 2026-02-18 09:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # users table
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("is_admin", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # images table
    op.create_table(
        "images",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("thumbnail_key", sa.String(512), nullable=True),
        sa.Column("content_type", sa.String(50), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("format", sa.String(10), nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_images_user_id", "images", ["user_id"])
    op.create_index(
        "idx_images_status",
        "images",
        ["status"],
        postgresql_where=sa.text("status != 'deleted'"),
    )

    # styles table
    op.create_table(
        "styles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_type", sa.String(30), nullable=False),
        sa.Column("model_key", sa.String(512), nullable=True),
        sa.Column("model_size_bytes", sa.Integer(), nullable=True),
        sa.Column("preview_key", sa.String(512), nullable=True),
        sa.Column("is_public", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("rating_sum", sa.Integer(), server_default="0", nullable=False),
        sa.Column("rating_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("download_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.String(20), server_default="active", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_styles_user_id", "styles", ["user_id"])
    op.create_index(
        "idx_styles_public",
        "styles",
        ["is_public", "created_at"],
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "idx_styles_model_type",
        "styles",
        ["model_type"],
        postgresql_where=sa.text("status = 'active'"),
    )

    # style_reference_images table
    op.create_table(
        "style_reference_images",
        sa.Column("style_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["style_id"], ["styles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("style_id", "image_id"),
    )

    # jobs table
    op.create_table(
        "jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), server_default="queued", nullable=False),
        sa.Column("progress", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("style_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("input_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("result_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("result_key", sa.String(512), nullable=True),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["input_image_id"], ["images.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["result_image_id"], ["images.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["style_id"], ["styles.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_jobs_user_id", "jobs", ["user_id", "created_at"])
    op.create_index(
        "idx_jobs_status",
        "jobs",
        ["status"],
        postgresql_where=sa.text("status IN ('queued', 'processing')"),
    )
    op.create_index("idx_jobs_style_id", "jobs", ["style_id"])
    op.create_index("idx_jobs_celery_task_id", "jobs", ["celery_task_id"])

    # comments table
    op.create_table(
        "comments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("style_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("length(content) > 0"),
        sa.ForeignKeyConstraint(["style_id"], ["styles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_comments_style_id",
        "comments",
        ["style_id", "created_at"],
        postgresql_where=sa.text("NOT is_deleted"),
    )

    # ratings table
    op.create_table(
        "ratings",
        sa.Column("style_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("score >= 1 AND score <= 5"),
        sa.ForeignKeyConstraint(["style_id"], ["styles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("style_id", "user_id"),
    )

    # refresh_tokens table
    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("idx_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index(
        "idx_refresh_tokens_expires",
        "refresh_tokens",
        ["expires_at"],
        postgresql_where=sa.text("revoked_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_refresh_tokens_expires", table_name="refresh_tokens")
    op.drop_index("idx_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_index("idx_comments_style_id", table_name="comments")
    op.drop_table("comments")
    op.drop_table("ratings")
    op.drop_index("idx_jobs_celery_task_id", table_name="jobs")
    op.drop_index("idx_jobs_style_id", table_name="jobs")
    op.drop_index("idx_jobs_status", table_name="jobs")
    op.drop_index("idx_jobs_user_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_table("style_reference_images")
    op.drop_index("idx_styles_model_type", table_name="styles")
    op.drop_index("idx_styles_public", table_name="styles")
    op.drop_index("idx_styles_user_id", table_name="styles")
    op.drop_table("styles")
    op.drop_index("idx_images_status", table_name="images")
    op.drop_index("idx_images_user_id", table_name="images")
    op.drop_table("images")
    op.drop_index("idx_users_email", table_name="users")
    op.drop_table("users")
