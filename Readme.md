# CS 579: Online Social Network Analysis

## Project 2 - Explainable graph neural network

### Team Members:

    Shubham Ganesh Modi ID: A20492276
    Oleksandr Shashkov ID: A20229995

### <b>Project Objectives</b>

    The goal of this project is to re-implement GNN Explaner described in[1], run experimental explanations on two different datasets not used in the original work, and analyze explanations obtained as a result of such experiments

<br>

### <b>GNN Explainer</b>

<br>

    The GNN Explainer used in this project is the explainer provided by the Torch Geometric library. The explainer has 2 functions Explain node and explaine graph.

    To explain the prediction for a particular node we use the explain node function.

    To visualize the output of the explainer which is a node features mask and an edge mask we call the visualize subgraph method of the explainer which takes in the nodes, labels and the edge masks and constructs a subgraph by calculating the number of hops the model takes to aggregate the information.

<br>

### <b>Task Distribution</b>

    Shubham worked on the Twitch dataset and did everything from Creating the GNN model, Training the model and Explaining the model.

    Oleksandr worked on the Cora dataset and did everything from Setting up the Dataset, training the model for his dataset and explaining the model.

### <b>Project Instructions</b>

To run the project, Follow the below steps

1.  Open the integrated command line and run the command <i>"pip install -r requirements.txt"</i>. This install are the required packages in the system.
2.  Open the <i><u>GNN Model Training.ipynb</u></i> Jupyter Notebook file and run the file as per the given instructions in the file. Choose an appropriate dataset while running the notebook {"Twitch", "Cora"}
3.  Open the <i><u>{model you choose in first part} GNN Explanations.ipynb</u></i> jupyter notebook file and run the file as per the given instructions in the file.
4.  Visualize the graphs and explain the nodes you'd like to understand the prediction for and display them.

<br>

### <b>Project Results</b>

<br>

<table border width=100%>
<tr>
<th colspan=6>
    <center>Dataset Training & Explainer Metrics
</th>
</tr>
<tr>
<th>
Dataset
</th>
<th>
    Training Accuracy
</th>
<th>
    Testing Accuracy
</th>
<th>
    Validation Accuracy
</th>
<th>
    Training Loss
</th>
<th>
    GNN Explainer ROC AUC Score
</th>
</tr>

<tr>
<th>
    Twitch
</th>
<td>
    0.7395
</td>
<td>
    0.7407
</td>
<td>
    0.7396
</td>
<td>
0.560089
</td>
<td>
0.68934
</td>
</tr>

<tr>
<th>
Cora
</th>
<td>
0.98
</td>
<td>
0.712
</td>
<td>
0.724
</td>
<td>
0.022679
</td>
<td>
0.7294
</td>
</tr>
</table>
