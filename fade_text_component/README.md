# FadeText Component

This is a custom Streamlit component that integrates a React-based `FadeText` component into a Streamlit app.

## Features
- Displays text with a fade-in or fade-out animation.
- Configurable animation direction and word delay.

## Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Build the component:
   ```bash
   npm run build
   ```

## Integration with Streamlit

Use the `FadeText` component in your Streamlit app by importing and using the component as follows:

```python
import streamlit as st
from streamlit_fadetext import st_fade_text

st_fade_text("Welcome to the future", direction="in", word_delay=0.2)
```