"""
Advanced data visualization utilities using matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
    """
    Create common data visualizations with sensible defaults.
    
    Examples:
        >>> viz = DataVisualizer()
        >>> viz.plot_distribution(data['age'], title='Age Distribution')
    """
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.fig_size = (10, 6)
    
    def plot_distribution(self, 
                         data: np.ndarray,
                         title: str = None,
                         xlabel: str = None,
                         ylabel: str = 'Frequency') -> None:
        """
        Plot distribution of numerical data.
        
        Args:
            data: Array-like data to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        plt.figure(figsize=self.fig_size)
        sns.histplot(data, kde=True)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()