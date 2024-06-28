import subprocess

def run_app():
    try:
        # Run add_ga.py and then start the Streamlit app
        subprocess.run("python add_ga.py && streamlit run main.py", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the app: {e}")

if __name__ == "__main__":
    run_app()
