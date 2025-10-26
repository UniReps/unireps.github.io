# Plotly Interactive Figures Directory

## How to Add Plotly Figures

1. Generate your plotly figure in Python
2. Save it as HTML using `fig.write_html('filename.html')`
3. Place the HTML file in this directory
4. Reference it in your blog post using:

```markdown
<div class="l-page">
  <iframe src="{{ '/assets/plotly/2025-10-16-why-did-the-ai-say-that/your-figure.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>
```

## Example Python Code

```python
import plotly.express as px
import pandas as pd

# Create your figure
fig = px.scatter(df, x='x', y='y', color='category')

# Save as HTML
fig.write_html('assets/plotly/2025-10-16-why-did-the-ai-say-that/my-figure.html')
```

