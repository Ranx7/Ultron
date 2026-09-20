import time
import os
import psutil

from sentence_transformers import SentenceTransformer

process = psutil.Process(os.getpid())

def stats():
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)

    return memory_mb, cpu_percent


print("Starting...")

# -------------------------
# Before model
# -------------------------
ram_before, cpu_before = stats()

start = time.time()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

load_time = time.time() - start

ram_after, cpu_after = stats()

print(f"\nMODEL LOAD: {load_time:.2f}s")
print(f"RAM BEFORE: {ram_before:.1f} MB")
print(f"RAM AFTER:  {ram_after:.1f} MB")
print(f"RAM USED:   {ram_after - ram_before:.1f} MB")
print(f"CPU:        {cpu_after:.1f}%")


# -------------------------
# Encoding
# -------------------------
ram_before, cpu_before = stats()

start = time.time()

embedding = model.encode("This is a test sentence.")

encode_time = time.time() - start

ram_after, cpu_after = stats()

print(f"\nENCODING: {encode_time:.2f}s")
print(f"RAM BEFORE: {ram_before:.1f} MB")
print(f"RAM AFTER:  {ram_after:.1f} MB")
print(f"RAM CHANGE: {ram_after - ram_before:.1f} MB")
print(f"CPU:        {cpu_after:.1f}%")