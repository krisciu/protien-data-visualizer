import time
import threading
#Temporary memory store as a workaround for no memory store for user feedback
#This should probably be replaced eventually
class TempMemoryStore:
    def __init__(self):
        self.store = {}
        self.lock = threading.Lock()

    def set(self, key, value, ttl=300):
        with self.lock:
            expiry = time.time() + ttl
            self.store[key] = (value, expiry)
            # Schedule cleanup for this specific key after ttl has passed
            threading.Timer(ttl, self.delete, args=[key]).start()

    def get(self, key):
        with self.lock:
            if key in self.store:
                value, expiry = self.store[key]
                if time.time() < expiry:
                    return value
                else:
                    # Entry has expired, delete it
                    self.delete(key)
        return None

    def delete(self, key):
        with self.lock:
            if key in self.store:
                del self.store[key]

# Create a global instance of the memory store
memory_store = TempMemoryStore()