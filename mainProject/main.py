from funcset import *
import streamlit as st
import subprocess

# This is the entire data exploration process.
cmd = subprocess.run('streamlit run webpage.py')

# This is the predict application based on webpage.
cmd2 = subprocess.run('streamlit run guiweb.py')