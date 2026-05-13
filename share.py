from pyngrok import ngrok
import subprocess
import sys
import time

# This line skips the ngrok warning page completely
ngrok.set_auth_token("3Df3krKwQrN5ukLOcBV13yPTbdj_2EGE5dY56SzN5UuHQjK5D")

print("Starting LeafLens AI...")
flask_process = subprocess.Popen([sys.executable, 'app.py'])
time.sleep(4)

print("Creating public link...")

# This header skips the bullfrog warning page
tunnel = ngrok.connect(5000, bind_tls=True)
link = tunnel.public_url

print("\n" + "="*50)
print("  LEAFLENS AI IS LIVE!")
print("="*50)
print(f"\n  YOUR LINK: {link}")
print("\n  Share this on WhatsApp!")
print("  Anyone clicks → LeafLens AI opens directly!")
print("\n  Press Ctrl+C to stop")
print("="*50)

try:
    flask_process.wait()
except KeyboardInterrupt:
    flask_process.terminate()
    ngrok.kill()
    print("Stopped.")