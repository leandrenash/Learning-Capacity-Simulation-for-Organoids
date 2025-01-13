# Organoid Learning Simulation Platform 

View Here - (https://organolearn.leandrenash.com/)

<img width="1457" alt="Screenshot 2025-01-13 at 19 32 00" src="https://github.com/user-attachments/assets/90ddcc25-c989-484f-a7ec-cd528d99d37a" />



## Overview
A sophisticated scientific simulation platform for analyzing organoid learning capacity through cutting-edge machine learning and interactive data visualization. This platform enables researchers to simulate and analyze how organoids learn and adapt in response to various stimuli, focusing on Organoid Intelligence (OI) and Organoid Learning (OL).

## 🧠 Key Features

### Organoid Intelligence Simulation
- Simulate organoid responses to external stimuli
- Model electrical pulses, chemical signals, and environmental changes
- Implement pattern recognition and decision-making tasks
- Real-time visualization of neural activity

### Stimulus-Response Modeling
- Apply various stimulus patterns:
  - Simple pulses
  - Complex patterns
  - Random sequences
- Track response latency and signal retention
- Analyze adaptability metrics

### Performance Analytics
- Learning curve visualization
- Memory retention tracking
- Response accuracy measurement
- Neural activity heatmaps
- Statistical analysis tools

## 🛠 Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly
- **Neural Simulation**: Custom OrganoidSimulation engine
- **Performance Metrics**: Custom analytics module

## 📊 Components

### OrganoidSimulation Class
The core simulation engine (`models/organoid_model.py`) provides:
- Neural network initialization
- Stimulus pattern generation
- State updates and response tracking
- Configurable parameters:
  - Number of neurons
  - Connectivity
  - Noise level

### Visualization Module
Interactive visualizations (`utils/visualization.py`):
- Neural activity heatmaps
- Learning curves
- Response patterns
- Real-time activity monitoring

### Analysis Tools
Comprehensive analytics (`utils/metrics.py`):
- Performance metrics calculation
- Learning efficiency
- Response accuracy
- Memory retention
- Statistical summaries

## 🚀 Getting Started

### Prerequisites
```bash
- Python 3.11 or higher
- Required packages:
  - streamlit
  - numpy
  - pandas
  - plotly
```

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run main.py
```

## 📝 Usage Guide

### Running Simulations
1. Navigate to the Simulation page
2. Configure parameters:
   - Number of neurons
   - Connectivity percentage
   - Noise level
   - Stimulus duration
3. Select stimulus pattern type
4. Click "Initialize/Run Simulation"

### Analyzing Results
1. View real-time visualizations:
   - Stimulus pattern
   - Network response
   - Neural activity heatmap
2. Access detailed analytics:
   - Performance metrics
   - Statistical summaries
   - Response comparisons

## 🔬 Example Workflow

1. **Setup Simulation**
   - Configure network parameters
   - Select stimulus pattern
   - Initialize simulation

2. **Monitor Response**
   - Observe neural activity
   - Track response patterns
   - Analyze adaptation

3. **Analyze Performance**
   - Review metrics
   - Export results
   - Compare patterns

## 📈 Future Developments

- Advanced stimulus patterns
- Enhanced visualization tools
- Export functionality
- Additional analysis metrics

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


For more information or support, please open an issue in the repository.
