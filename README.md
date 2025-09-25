# 🎬 Movie Booking API (FastAPI)

A demo movie ticket booking system built with **FastAPI**.  
This project was created as part of an assignment to design a backend API for movie theaters, halls, shows, and seat booking.

---

## 🚀 Features
- CRUD APIs for:
  - Movies
  - Theaters & Halls
  - Shows
  - Seats (booked / available)
- Seat booking for groups of friends (together, without gaps).
- If seats can’t be booked together → suggests other shows/movies.
- Prevents double booking with concurrent requests.
- Flexible hall layouts (rows, columns, aisles).
- Analytics API: GMV & tickets sold for a movie in a given period.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **FastAPI** (Web Framework)
- **SQLAlchemy** (ORM, SQLite DB)
- **Uvicorn** (ASGI server)

---

