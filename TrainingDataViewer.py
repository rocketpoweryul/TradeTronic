import logging
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import PreText

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def add_text(message):
    logging.info(message)
    curdoc().add_root(PreText(text=message))

def main():
    add_text("Starting application...")

    try:
        add_text("Attempting to load configuration...")
        # Simulate loading configuration
        add_text("Configuration loaded successfully.")

        add_text("Attempting to load data...")
        # Simulate loading data
        add_text("Data loaded successfully.")

        add_text("Setting up Bokeh elements...")
        # Simulate setting up Bokeh elements
        add_text("Bokeh elements set up successfully.")

        add_text("Application setup completed.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        add_text(error_message)

if __name__ == "__main__":
    main()