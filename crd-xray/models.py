from utils.base import Base
from sqlalchemy import Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime
from typing import Dict, Any

class CRD(Base):
    __tablename__ = "crd_table"

    crd: Mapped[str] = mapped_column(primary_key=True)
    last_updated_timestamp: Mapped[datetime] = mapped_column()
    controller_name: Mapped[str] = mapped_column()
    names: Mapped[Dict[str, Any]] = mapped_column(JSON)

    instances: Mapped[list["Instance"]] = relationship(
        back_populates="crd_obj", cascade="all, delete-orphan"
    )

class Instance(Base):
    __tablename__ = "instance_table"

    resource_name: Mapped[str] = mapped_column(primary_key=True)
    crd: Mapped[str] = mapped_column(ForeignKey("crd_table.crd"))
    crd_obj: Mapped["CRD"] = relationship(back_populates="instances")