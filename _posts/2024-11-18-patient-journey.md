---
layout: distill
title: Representing Patient Journeys in Healthcare with Large Language Models
tags: patient-journey llm healthcare representation-learning
giscus_comments: true
date: 2024-11-18
featured: true

authors:
  - name: Narmada Naik
    url: "https://www.datma.com/"
    affiliations:
      name: datma
  - name: Kevin Matlock
    url: "https://www.datma.com/"
    affiliations:
      name: datma

bibliography: 2024-11-18-patient-journey.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: "Case Study: Lung Cancer Dataset"
  - name: Representation and Embeddings
  - name: Goals of Analyzing Patient Journeys
  - name: Clustering
  - name: Results
  - name: Federated Patient Journey
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

Properly mapping out Patient Journeys in an healthcare system involves the analysis of both structured and unstructured dataset. In addition, this data is time-series in nature, making it difficult to apply standard clustering and prediction algorithms. This work explores a novel approach in mapping out Patient Journeys by combining Large Language Model (LLM) embeddings, K-Means clustering, and LLM based summarization. Embeddings are used to encode and project timeseries treatment regimes into a constant dimension vector, allowing K-Means to group journeys with similar patterns. Subsequently, an LLM summarizes the characteristics of each patient journey cluster, providing explainations on why patients are grouped together.  For instance, identifying clusters of patients with high readmission rates can help proactively address underlying issues. The resulting understanding of patient experiences can inform the development of targeted interventions and improve healthcare delivery.


---

## Case Study: Lung Cancer Dataset

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

---

## Goals of Analyzing Patient Journeys

### 1. Identifying Treatment Patterns  
- By studying patient journeys, researchers can uncover emerging patterns in treatment responses.  
- **Key outcomes include:**
  - Understanding which treatments work best for specific cancer types or patient subgroups.
  - Determining the optimal sequence of treatment administration to maximize efficacy.

### 2. Assessing Treatment Efficacy  
- Longitudinal data allows for evaluating the effectiveness of treatments over time.  
- **Crucial benefits include:**
  - Understanding long-term treatment outcomes.  
  - Identifying potential late-onset side effects.

### 3. Uncovering Patient Response Variability  
- Cancer treatments are personalized, with significant variability in patient responses.  
- **Insights gained:**
  - Factors influencing variability, such as **genetic markers**, **comorbidities**, or **socio-demographic factors**.  
  - Knowledge to develop more tailored treatment plans.

### 4. Improving Clinical Decision-Making  
- Patient journey data provides a detailed map for clinicians.  
- **Applications include:**
  - Comparing current patient data with historical records to predict outcomes.  
  - Adjusting treatment plans based on predicted trajectories.

### 5. Facilitating Clinical Research  
- Patient journeys are critical in the context of clinical trials.  
- **Use cases:**
  - Identifying eligible patients for trials.  
  - Tracking participant progress during trials.  
  - Analyzing trial data to derive meaningful conclusions.

---
## Representation and Embeddings


Our  method transforms raw patient data into informative embeddings that capture the  relationships between treatments, diagnostic events, genomic sequencing, and patient responses. The first step in our methodology involves extracting data into a structured format that includes event type, event description, and event date. The event type categorizes whether the event is therapeutic (e.g., drug treatment), diagnostic, or pertains to a change in the patient's condition. The event description provides detailed information about the event, such as drug names and dosages or a textual description of the diagnosis. The event date is used to chronologically order these events.
Using the event date, the events are organized into a series of Directed Acyclic Graphs (DAGs) for each patient, representing the sequence of discrete events they encounter. Each unique DAG is referred to as the "patient journey." These patient journeys often include repetitive event sequences, particularly for diagnostic purposes (e.g., consecutive "Condition" or "Panel" events), which can introduce noise into the data. To mitigate this noise, we employ an iterative filtering strategy:
### 1.Cleaning Event Description
Event descriptions are tokenized and standardized  for non-medication events replacing irrelevant terms with a curated set of common terms.
### 2.Identify Earliest Event 
For each patient, identify the earliest recorded event or treatment date.
### 3.Normalize Event Times
Compute the treatment day for each event by calculating the difference between the event date and the earliest event date. This normalizes all event times to a relative day count starting from the first treatment.
### 4.Aggregate Daily Events
For each treatment day, aggregate the drugs administered into a summary string. This concatenates the drugs administered on a given day into a single record.
### 5.Sort Events:
Ensure the data is sorted by patient and event time to maintain a chronological order of treatments.

Next, a text string representation of each patient's journey is generated. The summary for each patient includes:
-The treatment day.
-The drugs taken on that day.
-The diagnosis on that day.
-The diagnostic panel on that day.
These text strings can then be sent as input into a Large Language Model (LLM) to create a set of embeddings which can be used as a numerical representation for further analysis. These embeddings capture the semantic and syntactic nuances of the patient's journey, allowing us to compare and analyze different patient experiences.
Once generated, these embeddings are stored in a vector database, enabling efficient retrieval and similarity search. By clustering these embeddings, we can identify groups of patients with similar treatment trajectories, allowing for deeper insights into treatment effectiveness and potential side effects. This approach enables us to uncover hidden patterns and trends that may not be apparent through traditional data analysis methods.


---

## Clustering
 
This approach leverages pre-generated patient journey embeddings. These embeddings are numerical representations that capture the semantic relationships between words and events within a patient’s journey.The choice of embedding technique depends on the specific data structure (e.g., individual events vs. entire journey) and desired level of granularity. The K-Means algorithm uses a distance metric to determine the closest cluster centroid for each data point. Here, we performed Euclidean distance between two data points (embeddings) in the multidimensional space they occupy. The x-coordinate and y-coordinate represent the 2D projections of higher dimensional data, created by the t-SNE algorithm.Each dot corresponds to a specific patient’s data point. Patients are grouped based on similarities, which could be useful for further analysis like treatment effects, side effects, or disease progression.

Syntax highlighting is provided within `<d-code>` tags.
An example of inline code snippets: `<d-code language="html">let x = 10;</d-code>`.
For larger blocks of code, add a `block` attribute:

<d-code block language="javascript">
  var x = 25;
  function(x) {
    return x * x;
  }
</d-code>

**❗️ Note:** `<d-code>` blocks do not look good in the dark mode. ❗️

You can always use the default code-highlight using the `highlight` liquid tag:

{% highlight javascript %}
var x = 25;
function(x) {
return x \* x;
}
{% endhighlight %}

---
## Metrics
 Mutual Information Classification: We used a mutual information method to calculate the importance of each feature in separating data points into clusters. 
Cluster Quality Evaluation: Sum of Squared Errors (SSE): This metric SSE = σ∥x i − c j∥2 measures the total squared Euclidean distance between each data point and its assigned cluster centroid c j. 
Overall, this analysis combines feature importance with cluster quality metrics to evaluate the effectiveness of K-means clustering. 

## Results

You can add interative plots using plotly + iframes :framed_picture:

<!-- <div class="l-page">
  <iframe src="{{ '/assets/plotly/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div> -->
<div class="l-page">
  <iframe src="{{ '/assets/plotly/diagnostic.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>
<div class="l-page">
  <iframe src="{{ '/assets/plotly/treatment.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

The plot must be generated separately and saved into an HTML file.
To generate the plot that you see above, you can use the following code snippet:

{% highlight python %}
import pandas as pd
import plotly.express as px
df = pd.read_csv(
'https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv'
)
fig = px.density_mapbox(
df,
lat='Latitude',
lon='Longitude',
z='Magnitude',
radius=10,
center=dict(lat=0, lon=180),
zoom=0,
mapbox_style="stamen-terrain",
)
fig.show()
fig.write_html('assets/plotly/demo.html')
{% endhighlight %}

---

## Federated Patient Journey

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

---

## Conclusion

I am a conclusion