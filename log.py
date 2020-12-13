'''
log.py

Displays HTML-formatted output to facilitate troubleshooting.
'''

from IPython.core.display import display, HTML
import pandas

class Log:

    def __init__(self, disabled=False):
        '''
        The class constructor.
        '''
        self.disabled = disabled


    def dataframe(self, df):
        '''
        Prints a Pandas DataFrame to the window.

        Parameters:
            df (pd.DataFrame): the DataFrame

        Returns:
            None
        '''
        if not self.disabled:
            display(df)

    
    def detail(self, text):
        '''
        Prints a "p" HTML paragraph to the window.

        Parameters:
            text (str): the text

        Returns:
            None 
        '''
        self.info(text, "p")

    
    def header(self, text):
        '''
        Prints an "h5" HTML header to the window.

        Parameters:
            text (str): the text

        Returns:
            None
        '''
        self.info(text, "h5")


    def info(self, text, html_tag):
        '''
        Displays HTML-formatted text.

        Parameters:
            text (str): the text
            html_tag (str): the HTML tag

        Returns:
            None
        '''
        if not self.disabled:
            text_prefix = "-" if html_tag == "p" else ""
            flat_text = " ".join(text.splitlines())
            display(HTML(f"<{html_tag}>{text_prefix} {flat_text}</{html_tag}>"))

