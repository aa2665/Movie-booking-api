"""
fastapi_movie_booking.py

Single-file demo FastAPI app for Algo Bharat assignment.

Features included (minimal demo ready to push to GitHub):
- SQLite database (SQLAlchemy) - easy to run locally
- Models: Movie, Theater, Hall, Row, Seat, Show, ShowSeat, Booking, BookingSeat
- Endpoints: CRUD for movies/theaters/halls, create shows, get seat map, hold seats, confirm booking
- Simple contiguous-seat allocation (respects aisle positions stored per row)
- Optimistic locking via `version` on ShowSeat to avoid double-booking in concurrent confirm
- Seed function to create sample theater/hall/movies/shows

How to run locally:
1. Install dependencies:
   pip install fastapi uvicorn sqlalchemy pydantic

2. Run:
   python fastapi_movie_booking.py

3. Open docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import (create_engine, Column, Integer, String, ForeignKey, DateTime, Numeric, JSON,
                        Enum, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
import enum
import threading
import uuid
import time

DATABASE_URL = "sqlite:///./demo.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI(title="AlgoBharat Movie Booking - Demo")

# -----------------------------
# Database models
# -----------------------------

class BookingStatus(str, enum.Enum):
    pending = "pending"
    confirmed = "confirmed"
    cancelled = "cancelled"

class ShowSeatStatus(str, enum.Enum):
    available = "available"
    held = "held"
    booked = "booked"

class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    duration_mins = Column(Integer, nullable=True)
    description = Column(String, nullable=True)

class Theater(Base):
    __tablename__ = "theaters"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=True)
    halls = relationship("Hall", back_populates="theater")

class Hall(Base):
    __tablename__ = "halls"
    id = Column(Integer, primary_key=True)
    theater_id = Column(Integer, ForeignKey("theaters.id"), nullable=False)
    name = Column(String, nullable=False)
    theater = relationship("Theater", back_populates="halls")
    rows = relationship("Row", back_populates="hall")

class Row(Base):
    __tablename__ = "rows"
    id = Column(Integer, primary_key=True)
    hall_id = Column(Integer, ForeignKey("halls.id"), nullable=False)
    row_label = Column(String, nullable=False)
    seat_count = Column(Integer, nullable=False)
    # aisle_positions: list of seat indices after which aisle exists (1-based)
    aisle_positions = Column(JSON, default=[])  # e.g. [3,6]
    hall = relationship("Hall", back_populates="rows")
    seats = relationship("Seat", back_populates="row")

class Seat(Base):
    __tablename__ = "seats"
    id = Column(Integer, primary_key=True)
    hall_id = Column(Integer, ForeignKey("halls.id"), nullable=False)
    row_id = Column(Integer, ForeignKey("rows.id"), nullable=False)
    seat_index = Column(Integer, nullable=False)  # 1-based index
    seat_label = Column(String, nullable=False)   # e.g., A1
    row = relationship("Row", back_populates="seats")

class Show(Base):
    __tablename__ = "shows"
    id = Column(Integer, primary_key=True)
    hall_id = Column(Integer, ForeignKey("halls.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    base_price = Column(Numeric, nullable=False, default=100)
    show_seats = relationship("ShowSeat", back_populates="show")

class ShowSeat(Base):
    __tablename__ = "show_seats"
    id = Column(Integer, primary_key=True)
    show_id = Column(Integer, ForeignKey("shows.id"), nullable=False)
    seat_id = Column(Integer, ForeignKey("seats.id"), nullable=False)
    status = Column(String, nullable=False, default=ShowSeatStatus.available.value)
    hold_expires_at = Column(DateTime, nullable=True)
    # optimistic locking
    version = Column(Integer, nullable=False, default=0)
    show = relationship("Show", back_populates="show_seats")
    UniqueConstraint('show_id', 'seat_id', name='u_show_seat')

class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True)
    user_name = Column(String, nullable=False)
    user_contact = Column(String, nullable=True)
    show_id = Column(Integer, ForeignKey("shows.id"), nullable=False)
    total_amount = Column(Numeric, nullable=False, default=0)
    status = Column(String, nullable=False, default=BookingStatus.pending.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    seats = relationship("BookingSeat", back_populates="booking")

class BookingSeat(Base):
    __tablename__ = "booking_seats"
    id = Column(Integer, primary_key=True)
    booking_id = Column(Integer, ForeignKey("bookings.id"), nullable=False)
    show_seat_id = Column(Integer, ForeignKey("show_seats.id"), nullable=False)
    booking = relationship("Booking", back_populates="seats")

# Create tables
Base.metadata.create_all(bind=engine)

# -----------------------------
# Pydantic schemas
# -----------------------------
class MovieCreate(BaseModel):
    title: str
    duration_mins: Optional[int]
    description: Optional[str]

class TheaterCreate(BaseModel):
    name: str
    city: Optional[str]

class RowCreate(BaseModel):
    row_label: str
    seat_count: int = Field(..., ge=6)
    aisle_positions: Optional[List[int]] = []

class HallCreate(BaseModel):
    name: str
    rows: List[RowCreate]

class ShowCreate(BaseModel):
    movie_id: int
    start_time: datetime
    end_time: datetime
    base_price: float = 100.0

class SeatInfo(BaseModel):
    seat_id: int
    row: str
    index: int
    label: str
    status: str

class HoldRequest(BaseModel):
    show_id: int
    num_seats: int
    preferred_rows: Optional[List[str]] = None

class HoldResponse(BaseModel):
    hold_id: str
    seat_ids: List[int]
    expires_at: datetime

class ConfirmRequest(BaseModel):
    hold_id: str
    user_name: str
    user_contact: Optional[str] = None

class BookingResponse(BaseModel):
    booking_id: int
    status: str
    seat_ids: List[int]

# -----------------------------
# In-memory hold store (ephemeral) - for demo only
# hold_id -> {show_id, seat_ids, expires_at}
# thread-safe
# -----------------------------
holds_lock = threading.Lock()
holds: Dict[str, Dict[str, Any]] = {}
HOLD_TTL_SECONDS = 180  # 3 minutes

# background cleaner thread

def _hold_cleaner():
    while True:
        now = datetime.utcnow()
        to_delete = []
        with holds_lock:
            for hid, info in list(holds.items()):
                if info['expires_at'] <= now:
                    to_delete.append(hid)
            for hid in to_delete:
                # release show_seats held by this hold
                try:
                    db = SessionLocal()
                    for ssid in holds[hid]['show_seat_ids']:
                        ss = db.query(ShowSeat).filter(ShowSeat.id == ssid).first()
                        if ss and ss.status == ShowSeatStatus.held.value:
                            ss.status = ShowSeatStatus.available.value
                            ss.hold_expires_at = None
                            ss.version += 1
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()
                del holds[hid]
        time.sleep(5)

cleaner_thread = threading.Thread(target=_hold_cleaner, daemon=True)
cleaner_thread.start()

# -----------------------------
# Utility: seed data
# -----------------------------

def seed_sample():
    db = SessionLocal()
    try:
        # skip if movies exist
        if db.query(Movie).count() > 0:
            return
        m = Movie(title="Demo Movie", duration_mins=120, description="Sample movie")
        db.add(m)
        t = Theater(name="Demo Theater", city="Demo City")
        db.add(t)
        db.commit()
        # create hall with 3 rows
        h = Hall(theater_id=t.id, name="Hall 1")
        db.add(h)
        db.commit()
        rows_def = [
            ("A", 7, [3]),
            ("B", 8, [3,6]),
            ("C", 6, [3])
        ]
        for label, cnt, aisles in rows_def:
            r = Row(hall_id=h.id, row_label=label, seat_count=cnt, aisle_positions=aisles)
            db.add(r)
            db.commit()
            for i in range(1, cnt+1):
                s = Seat(hall_id=h.id, row_id=r.id, seat_index=i, seat_label=f"{label}{i}")
                db.add(s)
        db.commit()
        # create a show for the movie
        start = datetime.utcnow() + timedelta(minutes=30)
        show = Show(hall_id=h.id, movie_id=m.id, start_time=start, end_time=start + timedelta(hours=2), base_price=150)
        db.add(show)
        db.commit()
        # create show_seats
        seats = db.query(Seat).filter(Seat.hall_id == h.id).all()
        for s in seats:
            ss = ShowSeat(show_id=show.id, seat_id=s.id, status=ShowSeatStatus.available.value)
            db.add(ss)
        db.commit()
    except Exception as e:
        db.rollback()
        print("Seed error:", e)
    finally:
        db.close()

seed_sample()

# -----------------------------
# Seat allocation algorithm (per-row contiguous, no crossing aisles)
# -----------------------------

def find_contiguous_block(db, show_id: int, k: int, preferred_rows: Optional[List[str]] = None):
    # returns list of ShowSeat.ids or None
    # iterate rows in hall in preference order
    show = db.query(Show).filter(Show.id == show_id).first()
    if not show:
        return None
    # load rows for the hall
    rows = db.query(Row).filter(Row.hall_id == show.hall_id).all()
    # order rows: preferred first
    if preferred_rows:
        rows.sort(key=lambda r: (0 if r.row_label in preferred_rows else 1, r.row_label))
    else:
        rows.sort(key=lambda r: r.row_label)

    for row in rows:
        # get show seats for this row ordered by seat_index
        seats = db.query(Seat).filter(Seat.row_id == row.id).order_by(Seat.seat_index).all()
        # map seat_index -> show_seat
        index_to_ss = {}
        for s in seats:
            ss = db.query(ShowSeat).filter(ShowSeat.show_id == show_id, ShowSeat.seat_id == s.id).first()
            if not ss:
                continue
            index_to_ss[s.seat_index] = ss
        available_indices = [idx for idx, ss in index_to_ss.items() if ss.status == ShowSeatStatus.available.value]
        available_indices.sort()
        n = len(available_indices)
        if n < k:
            continue
        for i in range(0, n - k + 1):
            window = available_indices[i:i+k]
            if window[-1] - window[0] != k - 1:
                continue  # not strictly consecutive
            # ensure not crossing aisle
            crossing = False
            for a in row.aisle_positions or []:
                if a > window[0] and a <= window[-1]:
                    crossing = True
                    break
            if crossing:
                continue
            # found block
            ss_ids = [index_to_ss[idx].id for idx in window]
            return ss_ids
    return None

# -----------------------------
# API endpoints
# -----------------------------

@app.post("/movies", status_code=201)
def create_movie(payload: MovieCreate):
    db = SessionLocal()
    try:
        m = Movie(title=payload.title, duration_mins=payload.duration_mins, description=payload.description)
        db.add(m)
        db.commit()
        db.refresh(m)
        return {"id": m.id}
    finally:
        db.close()

@app.get("/movies")
def list_movies():
    db = SessionLocal()
    try:
        rows = db.query(Movie).all()
        return [{"id": r.id, "title": r.title} for r in rows]
    finally:
        db.close()

@app.post("/theaters", status_code=201)
def create_theater(p: TheaterCreate):
    db = SessionLocal()
    try:
        t = Theater(name=p.name, city=p.city)
        db.add(t)
        db.commit()
        db.refresh(t)
        return {"id": t.id}
    finally:
        db.close()

@app.post("/theaters/{theater_id}/halls", status_code=201)
def create_hall(theater_id: int, payload: HallCreate):
    db = SessionLocal()
    try:
        # validate theater
        th = db.query(Theater).filter(Theater.id == theater_id).first()
        if not th:
            raise HTTPException(status_code=404, detail="Theater not found")
        h = Hall(theater_id=theater_id, name=payload.name)
        db.add(h)
        db.commit()
        db.refresh(h)
        for r in payload.rows:
            row = Row(hall_id=h.id, row_label=r.row_label, seat_count=r.seat_count, aisle_positions=r.aisle_positions)
            db.add(row)
            db.commit()
            db.refresh(row)
            for i in range(1, r.seat_count + 1):
                seat_label = f"{r.row_label}{i}"
                s = Seat(hall_id=h.id, row_id=row.id, seat_index=i, seat_label=seat_label)
                db.add(s)
        db.commit()
        return {"id": h.id}
    finally:
        db.close()

@app.post("/halls/{hall_id}/shows", status_code=201)
def create_show(hall_id: int, payload: ShowCreate):
    db = SessionLocal()
    try:
        # validate hall & movie
        h = db.query(Hall).filter(Hall.id == hall_id).first()
        m = db.query(Movie).filter(Movie.id == payload.movie_id).first()
        if not h or not m:
            raise HTTPException(status_code=404, detail="Hall or Movie not found")
        show = Show(hall_id=hall_id, movie_id=payload.movie_id, start_time=payload.start_time, end_time=payload.end_time, base_price=payload.base_price)
        db.add(show)
        db.commit()
        db.refresh(show)
        # create show seats from hall seats
        seats = db.query(Seat).filter(Seat.hall_id == hall_id).all()
        for s in seats:
            ss = ShowSeat(show_id=show.id, seat_id=s.id, status=ShowSeatStatus.available.value)
            db.add(ss)
        db.commit()
        return {"id": show.id}
    finally:
        db.close()

@app.get("/shows/{show_id}/seats")
def get_show_seats(show_id: int):
    db = SessionLocal()
    try:
        show = db.query(Show).filter(Show.id == show_id).first()
        if not show:
            raise HTTPException(status_code=404, detail="Show not found")
        # collect seat info
        result: List[Dict[str, Any]] = []
        # join seats -> show_seats
        seats = db.query(Seat).join(ShowSeat, ShowSeat.seat_id == Seat.id).filter(ShowSeat.show_id == show_id).order_by(Seat.row_id, Seat.seat_index).all()
        for s in seats:
            ss = db.query(ShowSeat).filter(ShowSeat.show_id == show_id, ShowSeat.seat_id == s.id).first()
            row = db.query(Row).filter(Row.id == s.row_id).first()
            result.append({
                "seat_id": ss.id,
                "row": row.row_label,
                "index": s.seat_index,
                "label": s.seat_label,
                "status": ss.status
            })
        return {"show_id": show_id, "seats": result}
    finally:
        db.close()

# -----------------------------
# Hold seats endpoint
# -----------------------------
@app.post("/bookings/hold", response_model=HoldResponse)
def hold_seats(req: HoldRequest):
    db = SessionLocal()
    try:
        # basic validation
        show = db.query(Show).filter(Show.id == req.show_id).first()
        if not show:
            raise HTTPException(status_code=404, detail="Show not found")
        if req.num_seats < 1:
            raise HTTPException(status_code=400, detail="num_seats must be >= 1")
        ss_ids = find_contiguous_block(db, req.show_id, req.num_seats, req.preferred_rows)
        if not ss_ids:
            raise HTTPException(status_code=422, detail="Cannot find contiguous block for requested seats. Use suggestions endpoint.")
        # attempt to mark them as held (optimistic approach)
        locked_ids = []
        try:
            for ssid in ss_ids:
                ss = db.query(ShowSeat).filter(ShowSeat.id == ssid).with_for_update(nowait=True).first()
                if not ss:
                    raise HTTPException(status_code=500, detail="Internal error: show seat missing")
                if ss.status != ShowSeatStatus.available.value:
                    raise HTTPException(status_code=409, detail="Some seats are no longer available")
                ss.status = ShowSeatStatus.held.value
                ss.hold_expires_at = datetime.utcnow() + timedelta(seconds=HOLD_TTL_SECONDS)
                ss.version += 1
                locked_ids.append(ssid)
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=409, detail=f"Could not hold seats: {str(e)}")
        # create hold entry
        hid = str(uuid.uuid4())
        expires = datetime.utcnow() + timedelta(seconds=HOLD_TTL_SECONDS)
        with holds_lock:
            holds[hid] = {"show_id": req.show_id, "show_seat_ids": locked_ids, "expires_at": expires}
        return HoldResponse(hold_id=hid, seat_ids=locked_ids, expires_at=expires)
    finally:
        db.close()

# -----------------------------
# Confirm booking endpoint
# -----------------------------
@app.post("/bookings/confirm", response_model=BookingResponse)
def confirm_booking(req: ConfirmRequest):
    db = SessionLocal()
    try:
        with holds_lock:
            if req.hold_id not in holds:
                raise HTTPException(status_code=404, detail="Hold not found or expired")
            info = holds[req.hold_id]
            if info['expires_at'] <= datetime.utcnow():
                del holds[req.hold_id]
                raise HTTPException(status_code=410, detail="Hold expired")
            ss_ids = info['show_seat_ids']
        # finalize booking: verify all show_seats still held and update to booked atomically
        try:
            total = 0
            booking = Booking(user_name=req.user_name, user_contact=req.user_contact, show_id=info['show_id'], status=BookingStatus.pending.value)
            db.add(booking)
            db.commit()
            db.refresh(booking)
            for ssid in ss_ids:
                # reload and optimistic check
                ss = db.query(ShowSeat).filter(ShowSeat.id == ssid).first()
                if not ss or ss.status != ShowSeatStatus.held.value:
                    db.rollback()
                    raise HTTPException(status_code=409, detail="One or more seats are no longer held")
                # mark booked
                ss.status = ShowSeatStatus.booked.value
                ss.hold_expires_at = None
                ss.version += 1
                bs = BookingSeat(booking_id=booking.id, show_seat_id=ssid)
                db.add(bs)
                # price add
                s = db.query(Show).filter(Show.id == info['show_id']).first()
                total += float(s.base_price)
            booking.total_amount = total
            booking.status = BookingStatus.confirmed.value
            db.commit()
            # remove hold
            with holds_lock:
                if req.hold_id in holds:
                    del holds[req.hold_id]
            return BookingResponse(booking_id=booking.id, status=booking.status, seat_ids=ss_ids)
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Could not finalize booking: {str(e)}")
    finally:
        db.close()

# -----------------------------
# Simple suggestions endpoint (if cannot find contiguous seats)
# -----------------------------
@app.get("/shows/{show_id}/suggestions")
def show_suggestions(show_id: int, num_seats: int):
    db = SessionLocal()
    try:
        # try other shows same hall later/earlier within +/- 6 hours
        show = db.query(Show).filter(Show.id == show_id).first()
        if not show:
            raise HTTPException(status_code=404, detail="Show not found")
        window_start = show.start_time - timedelta(hours=6)
        window_end = show.start_time + timedelta(hours=6)
        candidates = db.query(Show).filter(Show.hall_id == show.hall_id, Show.start_time >= window_start, Show.start_time <= window_end, Show.id != show_id).all()
        out = []
        for c in candidates:
            block = find_contiguous_block(db, c.id, num_seats, None)
            if block:
                out.append({"show_id": c.id, "start_time": c.start_time.isoformat(), "seat_count": num_seats, "seat_ids": block})
        return {"suggestions": out}
    finally:
        db.close()

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
