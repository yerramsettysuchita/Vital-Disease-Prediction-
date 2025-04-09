import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
import pickle
import random
import sys  
import joblib

class HealthPredictorApp:
    def __init__(self, root):
        # Initialize main window
        self.root = root
        self.root.title("Health Predictor & Diet Recommendation System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set theme colors
        self.bg_color = "#f5f5f5"
        self.primary_color = "#4a7abc"
        self.secondary_color = "#2a9d8f"
        self.accent_color = "#e76f51"
        self.warning_color = "#e63946"
        self.success_color = "#43aa8b"
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'))
        self.style.configure('Accent.TButton', background=self.accent_color, foreground='white')
        self.style.configure('Primary.TButton', background=self.primary_color, foreground='white')
        self.style.configure('Success.TButton', background=self.success_color, foreground='white')
        self.style.configure('Heading.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Title.TLabel', font=('Arial', 18, 'bold'))
        
        # Initialize data variables
        self.df = None
        self.model = None
        self.scaler = None
        self.mlb = None
        self.data_file = "vital_disease_prediction_dataset.csv"
        self.model_file = "vital_disease_predictor.pkl"
        self.scaler_file = "scaler.pkl"
        self.mlb_file = "label_binarizer.pkl"
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize views (only one will be visible at a time)
        self.views = {
            "dashboard": self.create_dashboard_view(),
            "patients": self.create_patients_view(),
            "new_patient": self.create_new_patient_view(),
            "analysis": self.create_analysis_view(),
            "diet": self.create_diet_view()
        }
        
        # Load data and model
        self.load_data()
        self.load_model()
        
        # Show dashboard initially
        self.show_view("dashboard")
        
        # Update dashboard statistics
        self.update_dashboard()
    
    def create_sidebar(self):
        """Create the navigation sidebar"""
        sidebar = ttk.Frame(self.main_container, style='TFrame', width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)  # Prevent the sidebar from shrinking

        # Navigation buttons
        nav_buttons = [
            {"text": "Dashboard", "view": "dashboard", "icon": "📊"},
            {"text": "Patient Records", "view": "patients", "icon": "👤"},
            {"text": "New Patient", "view": "new_patient", "icon": "➕"},
            {"text": "Health Analysis", "view": "analysis", "icon": "📈"},
            {"text": "Diet Recommendations", "view": "diet", "icon": "🍎"}
        ]
        
        # Create a container for the navigation buttons
        nav_frame = ttk.Frame(sidebar, style='TFrame')
        nav_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Store button references
        self.nav_buttons = {}
        
        # Create custom buttons with custom style
        for i, btn_info in enumerate(nav_buttons):
            # Create a frame for button to style it better
            btn_frame = ttk.Frame(nav_frame, style='TFrame')
            btn_frame.pack(fill=tk.X, pady=5)
            
            # Add button
            button = tk.Button(
                btn_frame,
                text=f"{btn_info['icon']} {btn_info['text']}",
                font=('Arial', 11),
                bg=self.bg_color,
                fg="#333333",
                bd=0,
                padx=10,
                pady=8,
                anchor=tk.W,
                relief=tk.FLAT,
                overrelief=tk.FLAT,
                highlightthickness=0,
                wraplength=250,  # Add this to ensure text wraps
                width=25,  # Set a fixed width for the button
                command=lambda v=btn_info["view"]: self.show_view(v)
            )
            button.pack(fill=tk.X)
            
            # Store reference
            self.nav_buttons[btn_info["view"]] = button
        
        # Add divider
        divider = ttk.Separator(sidebar, orient=tk.HORIZONTAL)
        divider.pack(fill=tk.X, padx=10, pady=10)
        
        # Version info at bottom
        version_label = ttk.Label(
            sidebar,
            text="Version 1.0",
            foreground="gray",
            style='TLabel'
        )
        version_label.pack(side=tk.BOTTOM, pady=10)

        # Get current script directory to save files in the same location
        if getattr(sys, 'frozen', False):
            self.script_dir = os.path.dirname(sys.executable)
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(self.script_dir, "vital_disease_prediction_dataset.csv")
        self.model_file = os.path.join(self.script_dir, "vital_disease_predictor.pkl")
        self.scaler_file = os.path.join(self.script_dir, "scaler.pkl")
        self.mlb_file = os.path.join(self.script_dir, "label_binarizer.pkl")
    
    def show_view(self, view_name):
        """Show the selected view and hide others"""
        if view_name not in self.views:
            return
            
        # Update button styles
        for name, button in self.nav_buttons.items():
            if name == view_name:
                button.configure(bg=self.primary_color, fg="white")
            else:
                button.configure(bg=self.bg_color, fg="#333333")
        
        # Show selected view, hide others
        for name, frame in self.views.items():
            if name == view_name:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()
                
        # Additional actions when showing certain views
        if view_name == "dashboard":
            self.update_dashboard()
        elif view_name == "patients":
            self.display_patient_records(self.df)
        elif view_name == "analysis":
            self.update_analysis()
        elif view_name == "diet":
            self.update_diet_recommendations()
    
    def create_dashboard_view(self):
        """Create the dashboard view"""
        frame = ttk.Frame(self.content_frame, style='TFrame')
        
        # Header
        header = ttk.Label(
            frame, 
            text="Dashboard", 
            style='Title.TLabel'
        )
        header.pack(pady=(0, 20), anchor=tk.W)
        
        # Quick stats area with 4 cards
        stats_frame = ttk.Frame(frame, style='TFrame')
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Configure grid columns to be equal width
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(2, weight=1)
        stats_frame.columnconfigure(3, weight=1)
        
        # Create stat card frames
        self.stat_cards = []
        for i in range(4):
            card = tk.Frame(
                stats_frame,
                bg="white",
                highlightbackground="#e0e0e0",
                highlightthickness=1
            )
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            
            # Title label (will be populated later)
            title_label = ttk.Label(
                card,
                text="",
                background="white",
                font=('Arial', 10)
            )
            title_label.pack(anchor=tk.W, padx=15, pady=(15, 5))
            
            # Value label (will be populated later)
            value_label = ttk.Label(
                card,
                text="",
                background="white",
                font=('Arial', 20, 'bold')
            )
            value_label.pack(anchor=tk.W, padx=15, pady=(5, 15))
            
            self.stat_cards.append((title_label, value_label))
        
        # Charts area - two charts side by side
        charts_frame = ttk.Frame(frame, style='TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configure grid columns for charts
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Left chart (age distribution)
        self.age_chart_frame = tk.Frame(
            charts_frame,
            bg="white",
            highlightbackground="#e0e0e0",
            highlightthickness=1
        )
        self.age_chart_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Title for left chart
        ttk.Label(
            self.age_chart_frame,
            text="Age Distribution",
            background="white",
            font=('Arial', 12, 'bold')
        ).pack(anchor=tk.W, padx=15, pady=15)
        
        # Right chart (disease distribution)
        self.disease_chart_frame = tk.Frame(
            charts_frame,
            bg="white",
            highlightbackground="#e0e0e0",
            highlightthickness=1
        )
        self.disease_chart_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Title for right chart
        ttk.Label(
            self.disease_chart_frame,
            text="Top Diseases",
            background="white",
            font=('Arial', 12, 'bold')
        ).pack(anchor=tk.W, padx=15, pady=15)
        
        # Recent patients area
        recent_frame = tk.Frame(
            frame,
            bg="white",
            highlightbackground="#e0e0e0",
            highlightthickness=1
        )
        recent_frame.pack(fill=tk.X, pady=10)
        
        # Title for recent patients
        ttk.Label(
            recent_frame,
            text="Recent Patients",
            background="white",
            font=('Arial', 12, 'bold')
        ).pack(anchor=tk.W, padx=15, pady=15)
        
        # Table for recent patients
        self.recent_patients_frame = ttk.Frame(recent_frame, style='TFrame')
        self.recent_patients_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        return frame
        
    def update_dashboard(self):
        """Update the dashboard with current stats and charts"""
        if self.df is None or len(self.df) == 0:
            return
            
        # Update stat cards
        # Card 1: Total Patients
        self.stat_cards[0][0].config(text="Total Patients")
        self.stat_cards[0][1].config(text=str(len(self.df)))
        
        # Card 2: Healthy Patients
        healthy_count = len(self.df[self.df["Disease_Prediction"] == "None"])
        self.stat_cards[1][0].config(text="Healthy Patients")
        self.stat_cards[1][1].config(text=str(healthy_count))
        
        # Card 3: Patients with Conditions
        disease_count = len(self.df) - healthy_count
        self.stat_cards[2][0].config(text="With Conditions")
        self.stat_cards[2][1].config(text=str(disease_count))
        
        # Card 4: Most Common Disease
        all_diseases = []
        for diseases_str in self.df['Disease_Prediction'].dropna():
            if diseases_str != "None":
                all_diseases.extend(diseases_str.split(", "))
                
        if all_diseases:
            disease_counts = pd.Series(all_diseases).value_counts()
            most_common = disease_counts.index[0] if not disease_counts.empty else "None"
            most_common_count = disease_counts.values[0] if not disease_counts.empty else 0
            
            self.stat_cards[3][0].config(text="Most Common Disease")
            self.stat_cards[3][1].config(text=f"{most_common} ({most_common_count})")
        else:
            self.stat_cards[3][0].config(text="Most Common Disease")
            self.stat_cards[3][1].config(text="None")
        
        # Update age distribution chart
        self.update_age_chart()
        
        # Update disease distribution chart
        self.update_disease_chart()
        
        # Update recent patients list
        self.update_recent_patients()
        
    def update_age_chart(self):
        """Update the age distribution chart on dashboard"""
        # Clear existing chart
        for widget in self.age_chart_frame.winfo_children():
            if isinstance(widget, tk.Canvas) or isinstance(widget, FigureCanvasTkAgg):
                widget.destroy()
            
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Plot age histogram
        bins = np.linspace(10, 90, 9)  # Age bins from 10 to 90
        ax.hist(self.df['Age'], bins=bins, color=self.primary_color, alpha=0.7, edgecolor='white')
        
        # Add labels and title
        ax.set_xlabel('Age')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Age Distribution')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.age_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_disease_chart(self):
        """Update the disease distribution chart on dashboard"""
        # Clear existing chart
        for widget in self.disease_chart_frame.winfo_children():
            if isinstance(widget, tk.Canvas) or isinstance(widget, FigureCanvasTkAgg):
                widget.destroy()
            
        # Get disease counts
        all_diseases = []
        for diseases_str in self.df['Disease_Prediction'].dropna():
            if diseases_str != "None":
                all_diseases.extend(diseases_str.split(", "))
                
        if not all_diseases:
            # No diseases to display
            ttk.Label(
                self.disease_chart_frame,
                text="No disease data to display",
                background="white",
                font=('Arial', 12)
            ).pack(expand=True)
            return
            
        # Count occurrences of each disease
        disease_counts = pd.Series(all_diseases).value_counts().head(5)  # Top 5 diseases
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Create horizontal bar chart
        bars = ax.barh(
            disease_counts.index, 
            disease_counts.values,
            color=self.secondary_color,
            alpha=0.7
        )
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', 
                ha='left', 
                va='center'
            )
        
        # Add labels and title
        ax.set_xlabel('Number of Patients')
        ax.set_title('Top 5 Most Common Diseases')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.disease_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_recent_patients(self):
        """Update the recent patients list on dashboard"""
        # Clear existing list
        for widget in self.recent_patients_frame.winfo_children():
            widget.destroy()
            
        # Get most recent patients (last 5 in the dataframe)
        recent_df = self.df.tail(5)
        
        # Create header row
        ttk.Label(self.recent_patients_frame, text="Name", font=('Arial', 10, 'bold'), background="white").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        ttk.Label(self.recent_patients_frame, text="Age", font=('Arial', 10, 'bold'), background="white").grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        ttk.Label(self.recent_patients_frame, text="Gender", font=('Arial', 10, 'bold'), background="white").grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        ttk.Label(self.recent_patients_frame, text="Condition", font=('Arial', 10, 'bold'), background="white").grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
        
        # Add separator
        separator = ttk.Separator(self.recent_patients_frame, orient=tk.HORIZONTAL)
        separator.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=5)
        
        # Add patient rows
        for i, (_, row) in enumerate(recent_df.iterrows(), start=2):
            ttk.Label(self.recent_patients_frame, text=row['Name'], background="white").grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.recent_patients_frame, text=str(row['Age']), background="white").grid(row=i, column=1, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.recent_patients_frame, text=row['Gender'], background="white").grid(row=i, column=2, padx=10, pady=5, sticky=tk.W)
            
            if row['Disease_Prediction'] == "None":
                ttk.Label(
                    self.recent_patients_frame, 
                    text="Healthy", 
                    foreground="green",
                    background="white"
                ).grid(row=i, column=3, padx=10, pady=5, sticky=tk.W)
            else:
                ttk.Label(
                    self.recent_patients_frame, 
                    text=row['Disease_Prediction'], 
                    foreground="red",
                    background="white"
                ).grid(row=i, column=3, padx=10, pady=5, sticky=tk.W)
        
    def create_patients_view(self):
        """Create the patient records view"""
        frame = ttk.Frame(self.content_frame, style='TFrame')
    
        # Header with search options
        header_frame = ttk.Frame(frame, style='TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
    
        # Title
        ttk.Label(
            header_frame, 
            text="Patient Records", 
            style='Title.TLabel'
        ).pack(side=tk.LEFT)
    
        # Search box
        search_frame = ttk.Frame(header_frame, style='TFrame')
        search_frame.pack(side=tk.RIGHT)
    
        ttk.Label(search_frame, text="Search:", style='TLabel').pack(side=tk.LEFT, padx=(0, 5))
    
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=25)
        search_entry.pack(side=tk.LEFT, padx=(0, 5))
        search_entry.bind("<KeyRelease>", self.filter_patients)
    
        ttk.Button(
            search_frame,
            text="Search",
            command=self.filter_patients
        ).pack(side=tk.LEFT, padx=(0, 5))
    
        ttk.Button(
            search_frame,
            text="Clear",
            command=self.clear_patient_search
        ).pack(side=tk.LEFT)
    
        # Filter options
        filter_frame = ttk.Frame(frame, style='TFrame')
        filter_frame.pack(fill=tk.X, pady=(0, 10))
    
        # Disease filter
        ttk.Label(filter_frame, text="Filter by Disease:", style='TLabel').pack(side=tk.LEFT, padx=(0, 5))
    
        self.disease_filter_var = tk.StringVar()
        self.disease_filter = ttk.Combobox(
            filter_frame, 
            textvariable=self.disease_filter_var,
            state="readonly",
            width=20
        )
        self.disease_filter.pack(side=tk.LEFT, padx=(0, 5))
        self.disease_filter.bind("<<ComboboxSelected>>", self.filter_patients)
    
        # Gender filter
        ttk.Label(filter_frame, text="Gender:", style='TLabel').pack(side=tk.LEFT, padx=(15, 5))
    
        self.gender_filter_var = tk.StringVar()
        self.gender_filter = ttk.Combobox(
            filter_frame, 
            textvariable=self.gender_filter_var,
            values=["", "Male", "Female", "Other"],
            state="readonly",
            width=10
        )
        self.gender_filter.pack(side=tk.LEFT, padx=(0, 5))
        self.gender_filter.bind("<<ComboboxSelected>>", self.filter_patients)
      
        # Age range filter
        ttk.Label(filter_frame, text="Age Range:", style='TLabel').pack(side=tk.LEFT, padx=(15, 5))
    
        self.age_min_var = tk.StringVar(value="")
        age_min = ttk.Spinbox(
            filter_frame,
            from_=0,
            to=120,
            textvariable=self.age_min_var,
            width=5
        )
        age_min.pack(side=tk.LEFT)
    
        ttk.Label(filter_frame, text="-", style='TLabel').pack(side=tk.LEFT, padx=2)
    
        self.age_max_var = tk.StringVar(value="")
        age_max = ttk.Spinbox(
            filter_frame,
            from_=0,
            to=120,
            textvariable=self.age_max_var,
            width=5
        )
        age_max.pack(side=tk.LEFT, padx=(0, 5))
    
        ttk.Button(
            filter_frame,
            text="Apply Filters",
            command=self.filter_patients
        ).pack(side=tk.LEFT, padx=(5, 0))
    
        # Results count label
        self.results_label = ttk.Label(frame, text="", style='TLabel')
        self.results_label.pack(anchor=tk.W, pady=(0, 10))
    
        # Patient records area with scrollbar
        records_frame = ttk.Frame(frame, style='TFrame')
        records_frame.pack(fill=tk.BOTH, expand=True)
    
        # Create canvas with scrollbar for scrolling
        self.patient_canvas = tk.Canvas(records_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(records_frame, orient=tk.VERTICAL, command=self.patient_canvas.yview)
    
        # Configure scrolling
        self.patient_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.patient_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
        # Create frame inside canvas to hold patient cards
        self.patient_list_frame = ttk.Frame(self.patient_canvas, style='TFrame')
        self.patient_canvas_window = self.patient_canvas.create_window(
            (0, 0), window=self.patient_list_frame, anchor="nw"
        )
    
        # Configure canvas scrolling behavior
        self.patient_list_frame.bind(
            "<Configure>", 
            lambda e: self.patient_canvas.configure(scrollregion=self.patient_canvas.bbox("all"))
        )
        self.patient_canvas.bind(
            "<Configure>", 
            lambda e: self.patient_canvas.itemconfig(
                self.patient_canvas_window, width=e.width
            )
        )
        return frame

    def filter_patients(self, event=None):
        """Filter patient records based on search criteria"""
        if self.df is None:
            return
        
        filtered_df = self.df.copy()
        print("Starting patient filtering...")
        print(f"Total records before filtering: {len(filtered_df)}")
    
        # Apply name search
        name_filter = self.search_var.get().strip().lower()
        if name_filter:
            filtered_df = filtered_df[filtered_df['Name'].str.lower().str.contains(name_filter, na=False)]
            print(f"After name filter: {len(filtered_df)} records")
    
        # Apply disease filter
        disease_filter = self.disease_filter_var.get()
        if disease_filter:
            # Use case-insensitive search and handle NaN values
            # Need to use fillna because str.contains doesn't handle NaN values well
            filtered_df = filtered_df[filtered_df['Disease_Prediction'].fillna('None').str.contains(disease_filter, case=False, na=False)]
            print(f"After disease filter: {len(filtered_df)} records, filter value: '{disease_filter}'")
    
        # Apply gender filter
        gender_filter = self.gender_filter_var.get()
        if gender_filter:
            # Direct comparison for gender - handles both "Male"/"Female" and "M"/"F" formats
            gender_matches = filtered_df['Gender'].apply(
                lambda x: x.lower() == gender_filter.lower() or 
                          x.lower().startswith(gender_filter.lower()[0])
            )
            filtered_df = filtered_df[gender_matches]
            print(f"After gender filter: {len(filtered_df)} records, filter value: '{gender_filter}'")
    
         # Apply age range filter
        try:
            age_min = int(self.age_min_var.get()) if self.age_min_var.get() else None
            age_max = int(self.age_max_var.get()) if self.age_max_var.get() else None
        
            if age_min is not None:
                filtered_df = filtered_df[filtered_df['Age'] >= age_min]
                print(f"After min age filter: {len(filtered_df)} records, min age: {age_min}")
        
            if age_max is not None:
                filtered_df = filtered_df[filtered_df['Age'] <= age_max]
                print(f"After max age filter: {len(filtered_df)} records, max age: {age_max}")
            
        except ValueError as e:
            # Invalid age values, ignore filter
            print(f"Age filter error: {str(e)}")
            pass
    
        # Update results count
        self.results_label.config(text=f"Showing {len(filtered_df)} of {len(self.df)} patients")
    
        # Display filtered patients
        self.display_patient_records(filtered_df)

    
    def clear_patient_search(self):
        """Clear all patient search filters"""
        self.search_var.set("")
        self.disease_filter_var.set("")
        self.gender_filter_var.set("")
        self.age_min_var.set("")
        self.age_max_var.set("")
    
        # Reset results count
        self.results_label.config(text=f"Showing all {len(self.df)} patients")
    
        # Show all patients
        self.display_patient_records(self.df)

    def display_patient_records(self, filtered_df):
        """Display filtered patient records in the patients view"""
        # Clear existing patient cards
        for widget in self.patient_list_frame.winfo_children():
            widget.destroy()
    
        # Check if we have data to display
        if filtered_df is None or filtered_df.empty:
            ttk.Label(
                self.patient_list_frame,
                text="No patients found matching your criteria.",
                style='Heading.TLabel'
            ).pack(pady=50)
            return
    
    # Create patient cards - limit to first 100 for performance
        display_df = filtered_df.head(100) if len(filtered_df) > 100 else filtered_df
    
        for i, (idx, row) in enumerate(display_df.iterrows()):
            # Create card with border
            card = tk.Frame(
                self.patient_list_frame,
                bg="white",
                highlightbackground="#e0e0e0",
                highlightthickness=1
            )
            card.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        # Left side - Patient info
            left_frame = tk.Frame(card, bg="white")
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Patient name (bold)
            tk.Label(
                left_frame,
                text=row['Name'],
                font=('Arial', 12, 'bold'),
                bg="white",
                anchor=tk.W
            ).pack(fill=tk.X)
        
        # Basic info
            info_text = f"Age: {row['Age']} | Gender: {row['Gender']}"
            tk.Label(
                left_frame,
                text=info_text,
                font=('Arial', 10),
                bg="white",
                anchor=tk.W
            ).pack(fill=tk.X)
        
        # Right side - Disease and buttons
            right_frame = tk.Frame(card, bg="white")
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        
        # Disease status
            if pd.isna(row['Disease_Prediction']) or row['Disease_Prediction'] == "None":
                tk.Label(
                    right_frame,
                    text="Healthy",
                    font=('Arial', 10),
                    bg="white",
                    fg="green"
                ).pack(anchor=tk.E)
            else:
                tk.Label(
                    right_frame,
                    text=row['Disease_Prediction'],
                    font=('Arial', 10),
                    bg="white",
                    fg=self.warning_color,
                    wraplength=200,
                    justify=tk.RIGHT
                ).pack(anchor=tk.E)
        
        # View Details button
            view_btn = tk.Button(
                right_frame,
                text="View Details",
                font=('Arial', 9),
                bg=self.primary_color,
                fg="white",
                padx=10,
                relief=tk.FLAT,
                command=lambda r=row.copy(): self.view_patient_details(r)
            )
            view_btn.pack(anchor=tk.E, pady=5)
        
        # Show a message if there are more records
        if len(filtered_df) > 100:
            ttk.Label(
                self.patient_list_frame,
                text=f"Showing first 100 of {len(filtered_df)} matching records. Please add more filters to narrow results.",
                style='TLabel'
            ).pack(pady=10)

    def create_new_patient_view(self):
        """Create the new patient entry and prediction view"""
        frame = ttk.Frame(self.content_frame, style='TFrame')
        
        # Use Canvas with Scrollbar for the form
        canvas = tk.Canvas(frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a frame inside the canvas for the form
        form_frame = ttk.Frame(canvas, style='TFrame')
        canvas_window = canvas.create_window(
            (0, 0), window=form_frame, anchor="nw", width=canvas.winfo_width()
        )
        
        # Configure canvas scrolling behavior
        form_frame.bind(
            "<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", 
            lambda e: canvas.itemconfig(canvas_window, width=e.width)
        )
        
        # Title
        ttk.Label(
            form_frame, 
            text="New Patient", 
            style='Title.TLabel'
        ).pack(pady=(0, 20), anchor=tk.W)
        
        # Personal Information section
        personal_frame = ttk.LabelFrame(form_frame, text="Personal Information", style='TFrame')
        personal_frame.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        # Create a grid for personal info
        personal_grid = ttk.Frame(personal_frame, style='TFrame')
        personal_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Configure grid columns
        personal_grid.columnconfigure(0, weight=0)  # Label
        personal_grid.columnconfigure(1, weight=1)  # Field
        personal_grid.columnconfigure(2, weight=0)  # Label
        personal_grid.columnconfigure(3, weight=1)  # Field
        
        # Row 1: Name and Age
        ttk.Label(personal_grid, text="Full Name:", style='TLabel').grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        
        self.name_var = tk.StringVar()
        ttk.Entry(personal_grid, textvariable=self.name_var, width=30).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5
        )
        
        ttk.Label(personal_grid, text="Age:", style='TLabel').grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=5
        )
        
        self.age_var = tk.StringVar(value="30")
        ttk.Spinbox(personal_grid, from_=0, to=120, textvariable=self.age_var, width=10).grid(
            row=0, column=3, sticky=tk.W, padx=5, pady=5
        )
        
        # Row 2: Gender and any additional personal info you want
        ttk.Label(personal_grid, text="Gender:", style='TLabel').grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        
        self.gender_var = tk.StringVar(value="Male")
        ttk.Combobox(
            personal_grid, 
            textvariable=self.gender_var,
            values=["Male", "Female", "Other"],
            state="readonly",
            width=10
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Vital Health Metrics section
        vitals_frame = ttk.LabelFrame(form_frame, text="Vital Health Metrics", style='TFrame')
        vitals_frame.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        # Create a grid for vitals
        vitals_grid = ttk.Frame(vitals_frame, style='TFrame')
        vitals_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Configure grid columns
        for i in range(6):
            vitals_grid.columnconfigure(i, weight=1)
        
        # Define vital signs with default values and ranges
        self.vital_signs = [
            {"name": "Hemoglobin", "unit": "g/dL", "default": 14.0, "min": 5.0, "max": 20.0, "row": 0, "col": 0, "var_name": "hemoglobin"},
            {"name": "BP Systolic", "unit": "mmHg", "default": 120, "min": 70, "max": 200, "row": 0, "col": 2, "var_name": "bp_systolic"},
            {"name": "BP Diastolic", "unit": "mmHg", "default": 80, "min": 40, "max": 120, "row": 0, "col": 4, "var_name": "bp_diastolic"},
            {"name": "Heart Rate", "unit": "bpm", "default": 75, "min": 40, "max": 150, "row": 1, "col": 0, "var_name": "heart_rate"},
            {"name": "HbA1c", "unit": "%", "default": 5.5, "min": 4.0, "max": 10.0, "row": 1, "col": 2, "var_name": "hba1c"},
            {"name": "Vitamin D", "unit": "ng/mL", "default": 30.0, "min": 10.0, "max": 60.0, "row": 1, "col": 4, "var_name": "vitamin_d"},
            {"name": "LDL", "unit": "mg/dL", "default": 100.0, "min": 50.0, "max": 200.0, "row": 2, "col": 0, "var_name": "ldl"},
            {"name": "Iron", "unit": "μg/dL", "default": 100.0, "min": 30.0, "max": 180.0, "row": 2, "col": 2, "var_name": "iron"},
            {"name": "Creatinine", "unit": "mg/dL", "default": 1.0, "min": 0.5, "max": 3.0, "row": 2, "col": 4, "var_name": "creatinine"},
            {"name": "CRP", "unit": "mg/L", "default": 2.0, "min": 0.0, "max": 50.0, "row": 3, "col": 0, "var_name": "crp"},
            {"name": "MCH", "unit": "", "default": 29.0, "min": 20.0, "max": 40.0, "row": 3, "col": 2, "var_name": "mch"},
            {"name": "MCHC", "unit": "", "default": 33.0, "min": 30.0, "max": 38.0, "row": 3, "col": 4, "var_name": "mchc"},
        ]
        
        # Create fields for vital signs
        self.vital_vars = {}
        
        for vital in self.vital_signs:
            # Label with unit
            label_text = f"{vital['name']} ({vital['unit']})" if vital['unit'] else vital['name']
            ttk.Label(vitals_grid, text=label_text + ":", style='TLabel').grid(
                row=vital["row"], column=vital["col"], sticky=tk.W, padx=5, pady=5
            )
            
            # Input field with variable
            self.vital_vars[vital["var_name"]] = tk.DoubleVar(value=vital["default"])
            ttk.Spinbox(
                vitals_grid,
                from_=vital["min"],
                to=vital["max"],
                textvariable=self.vital_vars[vital["var_name"]],
                width=8,
                increment=0.1
            ).grid(row=vital["row"], column=vital["col"]+1, sticky=tk.W, padx=5, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(form_frame, style='TFrame')
        button_frame.pack(fill=tk.X, pady=15)
        
        ttk.Button(
            button_frame,
            text="Predict Disease",
            style='Primary.TButton',
            command=self.predict_disease
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Form",
            command=self.clear_patient_form
        ).pack(side=tk.LEFT, padx=5)
        
        # Results area
        self.prediction_frame = ttk.LabelFrame(
            form_frame, 
            text="Prediction Results", 
            style='TFrame'
        )
        
        # Diet recommendations area
        self.diet_recommendation_frame = ttk.LabelFrame(
            form_frame, 
            text="Diet Recommendations", 
            style='TFrame'
        )
        
        return frame
    
    def predict_disease(self):
        """Make disease prediction based on input values"""
    # Validate required fields
        if not self.name_var.get().strip():
            messagebox.showerror("Validation Error", "Please enter the patient's name")
            return
        
        try:
            age = int(self.age_var.get())
            if age <= 0 or age > 120:
                raise ValueError("Invalid age")
        except ValueError:
            messagebox.showerror("Validation Error", "Please enter a valid age (1-120)")
            return
        
    # Collect vital signs
        vital_values = {}
        for name, var in self.vital_vars.items():
            vital_values[name] = var.get()
        
    # If no model, show error
        if self.model is None or self.scaler is None or self.mlb is None:
            messagebox.showerror("Model Error", "Prediction model not loaded. Please train the model first.")
            return
        
        try:
            if not hasattr(self.scaler, 'transform'):
                messagebox.showerror("Model Error", "The scaler is not properly loaded. Retraining the model.")
                self.train_model()
                return
        
        # Prepare feature vector in the correct order for the model
            feature_cols = [col for col in self.df.columns if col not in ["Name", "Gender", "Disease_Prediction"]]
        
        # Map input vital values to feature columns
            features = []
            for col in feature_cols:
                if col == "Age":
                    features.append(age)
                else:
                    var_name = col.lower().replace('_', '').lower()
                    matching_vars = [v for k, v in vital_values.items() if k.lower().replace('_', '') == var_name]
                    if matching_vars:
                        features.append(matching_vars[0])
                    else:
                        features.append(0.0)
        
        # Convert features to numpy array and reshape if needed
            features = np.array(features).reshape(1, -1)
            print(f"Feature shape: {features.shape}, Feature columns: {feature_cols}")
        
        # Scale the features
            scaled_features = self.scaler.transform(features)
        
        # Make prediction
            prediction = self.model.predict(scaled_features)
        
        # Convert prediction to disease names
            predicted_diseases = []
            for i, val in enumerate(prediction[0]):
                if val == 1 and i < len(self.mlb.classes_):
                    predicted_diseases.append(self.mlb.classes_[i])
        
        # Get probabilities
            disease_probs = []
            for i, disease in enumerate(self.mlb.classes_):
                if hasattr(self.model, 'estimators_') and i < len(self.model.estimators_):
                    estimator = self.model.estimators_[i]
                    if hasattr(estimator, 'predict_proba'):
                        prob = estimator.predict_proba(scaled_features)[0][1]
                        disease_probs.append((disease, prob))
        
        # Sort diseases by probability
            disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Display prediction results
            self.display_prediction_results(predicted_diseases, disease_probs)
        
        # Generate diet recommendations
            self.display_diet_recommendations(predicted_diseases)
        
        # Save patient data
            self.save_patient_data(predicted_diseases)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Prediction Error: {str(e)}\n{error_details}")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {str(e)}")
    
    def display_prediction_results(self, predicted_diseases, disease_probs):
        """Display disease prediction results"""
        # Clear any existing prediction frame
        if self.prediction_frame.winfo_manager():
            self.prediction_frame.pack_forget()
            
        # Pack the frame
        self.prediction_frame.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        # Clear existing content
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
            
        # Display results
        if not predicted_diseases:
            # No diseases predicted
            result_label = ttk.Label(
                self.prediction_frame,
                text="No health conditions detected. The patient appears to be healthy.",
                font=('Arial', 12),
                foreground="green",
                style='TLabel'
            )
            result_label.pack(padx=15, pady=15)
        else:
            # Create scrollable text area for results
            result_text = scrolledtext.ScrolledText(
                self.prediction_frame, 
                height=10, 
                wrap=tk.WORD, 
                font=('Arial', 11)
            )
            result_text.pack(fill=tk.BOTH, padx=15, pady=15)
            
            # Insert header
            result_text.insert(tk.END, "Predicted Health Conditions:\n\n", "header")
            result_text.tag_configure("header", font=('Arial', 12, 'bold'))
            
            # Insert each disease with probability
            for disease, prob in disease_probs:
                if prob > 0.3:  # Only show diseases with >30% probability
                    line = f"• {disease}: {prob*100:.1f}% probability\n"
                    result_text.insert(tk.END, line, "disease" if prob > 0.5 else "normal")
                    
            # Configure tags
            result_text.tag_configure("disease", foreground=self.warning_color, font=('Arial', 11, 'bold'))
            result_text.tag_configure("normal", font=('Arial', 11))
            
            # Add recommendations
            result_text.insert(tk.END, "\nRecommendations:\n", "header")
            recommendations = self.generate_recommendations(predicted_diseases)
            result_text.insert(tk.END, recommendations)
            
            # Make read-only
            result_text.config(state=tk.DISABLED)
    
    def generate_recommendations(self, diseases):
        """Generate health recommendations based on predicted diseases"""
        recommendations = ""
        
        if "Anemia" in diseases:
            recommendations += "• Consider iron supplements and include iron-rich foods in diet\n"
            recommendations += "• Schedule follow-up blood tests to monitor hemoglobin levels\n"
            
        if "Hypertension" in diseases:
            recommendations += "• Monitor blood pressure regularly\n"
            recommendations += "• Reduce sodium intake and maintain a heart-healthy diet\n"
            recommendations += "• Consider medication if lifestyle changes don't improve readings\n"
            
        if "Diabetes" in diseases:
            recommendations += "• Monitor blood glucose levels regularly\n"
            recommendations += "• Follow a controlled carbohydrate diet plan\n"
            recommendations += "• Consider consultation with an endocrinologist\n"
            
        if "Heart Disease" in diseases:
            recommendations += "• Schedule an appointment with a cardiologist\n"
            recommendations += "• Consider cardiac stress test and additional evaluations\n"
            recommendations += "• Follow a heart-healthy diet low in saturated fats\n"
            
        if "Vitamin D Deficiency" in diseases:
            recommendations += "• Take vitamin D supplements as directed\n"
            recommendations += "• Increase sun exposure (15-20 minutes daily if possible)\n"
            
        if "Kidney Disease" in diseases:
            recommendations += "• Consult a nephrologist for specialized care\n"
            recommendations += "• Monitor fluid intake and follow a kidney-friendly diet\n"
            recommendations += "• Schedule regular kidney function tests\n"
            
        if "High Cholesterol" in diseases:
            recommendations += "• Follow a low-cholesterol diet\n"
            recommendations += "• Increase physical activity\n"
            recommendations += "• Consider medication if levels remain elevated\n"
            
        if not recommendations:
            recommendations = "Maintain a balanced diet and regular exercise routine for overall health.\n"
            recommendations += "Schedule annual check-ups to monitor health status.\n"
            
        return recommendations
    
    def clear_patient_form(self):
        """Clear all fields in the new patient form"""
        # Clear name
        self.name_var.set("")
        
        # Reset age to default
        self.age_var.set("30")
        
        # Reset gender to default
        self.gender_var.set("Male")
        
        # Reset all vital signs to defaults
        for vital in self.vital_signs:
            self.vital_vars[vital["var_name"]].set(vital["default"])
            
        # Hide results frames if visible
        if self.prediction_frame.winfo_manager():
            self.prediction_frame.pack_forget()
            
        if self.diet_recommendation_frame.winfo_manager():
            self.diet_recommendation_frame.pack_forget()
            
        messagebox.showinfo("Form Cleared", "All fields have been reset to default values.")

    def create_analysis_view(self):
        """Create the health analysis view with interactive charts"""
        frame = ttk.Frame(self.content_frame, style='TFrame')
        
        # Title
        ttk.Label(
            frame, 
            text="Health Analysis", 
            style='Title.TLabel'
        ).pack(pady=(0, 20), anchor=tk.W)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Disease Distribution
        self.disease_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(self.disease_tab, text="Disease Analysis")
        
        # Tab 2: Vital Signs Analysis
        self.vitals_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(self.vitals_tab, text="Vital Signs")
        
        # Tab 3: Age & Gender Analysis
        self.demographics_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(self.demographics_tab, text="Demographics")
        
        # Tab 4: Correlation Analysis
        self.correlation_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(self.correlation_tab, text="Correlations")
        
        # Setup each tab
        self.setup_disease_tab()
        self.setup_vitals_tab()
        self.setup_demographics_tab()
        self.setup_correlation_tab()
        
        return frame
    
    def setup_disease_tab(self):
        """Setup the disease analysis tab"""
        # Top frame for disease frequency chart
        top_frame = ttk.Frame(self.disease_tab, style='TFrame')
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(
            top_frame, 
            text="Disease Frequency", 
            style='Heading.TLabel'
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Create frame for chart
        self.disease_freq_chart_frame = ttk.Frame(top_frame, style='TFrame')
        self.disease_freq_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame for disease co-occurrence
        bottom_frame = ttk.Frame(self.disease_tab, style='TFrame')
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(
            bottom_frame, 
            text="Disease Co-occurrence", 
            style='Heading.TLabel'
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Create frame for chart
        self.disease_cooccur_chart_frame = ttk.Frame(bottom_frame, style='TFrame')
        self.disease_cooccur_chart_frame.pack(fill=tk.BOTH, expand=True)
    
    def setup_vitals_tab(self):
        """Setup the vital signs analysis tab"""
        # Controls frame
        controls_frame = ttk.Frame(self.vitals_tab, style='TFrame')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Dropdown for selecting vital sign
        ttk.Label(controls_frame, text="Select Vital Sign:", style='TLabel').pack(side=tk.LEFT, padx=(0, 5))
        
        self.selected_vital = tk.StringVar()
        vital_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.selected_vital,
            state="readonly",
            width=15
        )
        vital_dropdown.pack(side=tk.LEFT, padx=5)
        vital_dropdown.bind("<<ComboboxSelected>>", self.update_vital_chart)
        
        # Chart frame
        self.vital_chart_frame = ttk.Frame(self.vitals_tab, style='TFrame')
        self.vital_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics frame
        self.vital_stats_frame = ttk.Frame(self.vitals_tab, style='TFrame')
        self.vital_stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # If we have data, populate the dropdown
        if self.df is not None:
            numeric_cols = [col for col in self.df.columns 
                          if col not in ["Name", "Gender", "Disease_Prediction"]
                          and pd.api.types.is_numeric_dtype(self.df[col])]
            
            vital_dropdown['values'] = numeric_cols
            if numeric_cols:
                self.selected_vital.set(numeric_cols[0])
                self.update_vital_chart()
    
    def setup_demographics_tab(self):
        """Setup the demographics analysis tab"""
        # Left frame for age distribution
        left_frame = ttk.Frame(self.demographics_tab, style='TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(
            left_frame, 
            text="Age Distribution", 
            style='Heading.TLabel'
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Chart frame for age
        self.age_demo_chart_frame = ttk.Frame(left_frame, style='TFrame')
        self.age_demo_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Right frame for gender distribution
        right_frame = ttk.Frame(self.demographics_tab, style='TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(
            right_frame, 
            text="Gender Distribution", 
            style='Heading.TLabel'
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Chart frame for gender
        self.gender_chart_frame = ttk.Frame(right_frame, style='TFrame')
        self.gender_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Update charts if data is available
        if self.df is not None and len(self.df) > 0:
            self.update_demographics_analysis()
    
    def setup_correlation_tab(self):
        """Setup the correlation analysis tab"""
        # Title
        ttk.Label(
            self.correlation_tab, 
            text="Correlation Between Health Metrics", 
            style='Heading.TLabel'
        ).pack(anchor=tk.W, padx=10, pady=10)
        
        # Chart frame for correlation matrix
        self.correlation_chart_frame = ttk.Frame(self.correlation_tab, style='TFrame')
        self.correlation_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Update correlation chart if data is available
        if self.df is not None and len(self.df) > 0:
            self.update_correlation_analysis()
            
    def update_analysis(self):
        """Update all analysis charts when analysis view is shown"""
        if self.df is None or len(self.df) == 0:
            return
            
        # Update disease frequency chart
        self.update_disease_frequency_chart()
        
        # Update disease co-occurrence chart
        self.update_disease_cooccurrence_chart()
        
        # Update vital sign chart
        self.update_vital_chart()
        
        # Update demographics charts
        self.update_demographics_analysis()
        
        # Update correlation analysis
        self.update_correlation_analysis()
        
    def update_disease_frequency_chart(self):
        """Update the disease frequency chart in analysis view"""
        # Clear existing chart
        for widget in self.disease_freq_chart_frame.winfo_children():
            widget.destroy()
            
        # Get disease counts
        all_diseases = []
        for diseases_str in self.df['Disease_Prediction'].dropna():
            if diseases_str != "None":
                all_diseases.extend(diseases_str.split(", "))
                
        if not all_diseases:
            # No diseases to display
            ttk.Label(
                self.disease_freq_chart_frame,
                text="No disease data to display",
                font=('Arial', 12)
            ).pack(expand=True)
            return
            
        # Count occurrences of each disease
        disease_counts = pd.Series(all_diseases).value_counts()
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = ax.barh(
            disease_counts.index, 
            disease_counts.values,
            color=self.primary_color,
            alpha=0.7
        )
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', 
                ha='left', 
                va='center'
            )
        
        # Add labels and title
        ax.set_xlabel('Number of Patients')
        ax.set_title('Disease Frequency Distribution')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.disease_freq_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_disease_cooccurrence_chart(self):
        """Update the disease co-occurrence chart in analysis view"""
        # Clear existing chart
        for widget in self.disease_cooccur_chart_frame.winfo_children():
            widget.destroy()
            
        # Get all diseases
        all_diseases = set()
        disease_lists = []
        
        for diseases_str in self.df['Disease_Prediction'].dropna():
            if diseases_str != "None":
                disease_list = diseases_str.split(", ")
                disease_lists.append(disease_list)
                all_diseases.update(disease_list)
                
        all_diseases = list(all_diseases)
        
        if not all_diseases:
            # No diseases to display
            ttk.Label(
                self.disease_cooccur_chart_frame,
                text="No disease data to display",
                font=('Arial', 12)
            ).pack(expand=True)
            return
            
        # Create co-occurrence matrix
        cooccur_matrix = np.zeros((len(all_diseases), len(all_diseases)))
        
        for disease_list in disease_lists:
            for i, disease1 in enumerate(all_diseases):
                if disease1 in disease_list:
                    for j, disease2 in enumerate(all_diseases):
                        if disease2 in disease_list:
                            cooccur_matrix[i, j] += 1
                            
        # Create DataFrame for heatmap
        cooccur_df = pd.DataFrame(cooccur_matrix, index=all_diseases, columns=all_diseases)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cooccur_df,
            annot=True,
            cmap="YlGnBu",
            ax=ax,
            fmt=".0f"
        )
        
        # Add title
        ax.set_title('Disease Co-occurrence Matrix')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.disease_cooccur_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_vital_chart(self, event=None):
        """Update the vital sign distribution chart"""
        if self.df is None or len(self.df) == 0 or not self.selected_vital.get():
            return
            
        vital = self.selected_vital.get()
        
        # Clear existing chart
        for widget in self.vital_chart_frame.winfo_children():
            widget.destroy()
            
        # Clear existing stats
        for widget in self.vital_stats_frame.winfo_children():
            widget.destroy()
            
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with KDE
        sns.histplot(
            data=self.df,
            x=vital,
            kde=True,
            color=self.primary_color,
            ax=ax
        )
        
        # Add reference lines for normal ranges if applicable
        normal_range = self.get_normal_range(vital)
        range_parts = normal_range.replace('(M)', '').replace('(F)', '').split('-')
        
        try:
            if len(range_parts) == 2:
                low = float(range_parts[0].strip().replace('<', '').replace('>', ''))
                high = float(range_parts[1].strip().replace('<', '').replace('>', ''))
                
                ax.axvline(x=low, color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=high, color='red', linestyle='--', alpha=0.7)
                
                # Add shaded normal range area
                ax.axvspan(low, high, alpha=0.2, color='green')
            elif '<' in normal_range:
                # Handle ranges like "<5.7%"
                value = float(normal_range.replace('<', '').strip().replace('%', ''))
                ax.axvline(x=value, color='red', linestyle='--', alpha=0.7)
            elif '>' in normal_range:
                # Handle ranges like ">30 ng/mL"
                value = float(normal_range.replace('>', '').strip().split()[0])
                ax.axvline(x=value, color='red', linestyle='--', alpha=0.7)
        except (ValueError, IndexError):
            # If we can't parse the normal range, just skip the reference lines
            pass
            
        # Add labels and title
        ax.set_xlabel(vital)
        ax.set_ylabel('Number of Patients')
        ax.set_title(f'Distribution of {vital}')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.vital_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add statistics in the bottom frame
        self.display_vital_statistics(vital)
    
    def display_vital_statistics(self, vital):
        """Display statistics for the selected vital sign"""
        # Calculate statistics
        mean_val = self.df[vital].mean()
        median_val = self.df[vital].median()
        std_val = self.df[vital].std()
        min_val = self.df[vital].min()
        max_val = self.df[vital].max()
        
        # Create statistics grid
        stats_frame = ttk.Frame(self.vital_stats_frame, style='TFrame')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add statistics labels
        ttk.Label(stats_frame, text="Mean:", style='TLabel', font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, text=f"{mean_val:.2f}", style='TLabel').grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Median:", style='TLabel', font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, text=f"{median_val:.2f}", style='TLabel').grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Std Dev:", style='TLabel', font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=10, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, text=f"{std_val:.2f}", style='TLabel').grid(row=0, column=5, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Min:", style='TLabel', font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, text=f"{min_val:.2f}", style='TLabel').grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Max:", style='TLabel', font=('Arial', 10, 'bold')).grid(row=1, column=2, padx=10, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, text=f"{max_val:.2f}", style='TLabel').grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Normal Range:", style='TLabel', font=('Arial', 10, 'bold')).grid(row=1, column=4, padx=10, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, text=self.get_normal_range(vital), style='TLabel').grid(row=1, column=5, padx=10, pady=5, sticky=tk.W)
        
        # Add abnormal count
        abnormal_count = self.count_abnormal_values(vital)
        if abnormal_count > 0:
            ttk.Label(
                stats_frame, 
                text=f"Abnormal Values: {abnormal_count} patients ({abnormal_count / len(self.df) * 100:.1f}%)",
                style='TLabel',
                font=('Arial', 10),
                foreground=self.warning_color
            ).grid(row=2, column=0, columnspan=6, padx=10, pady=10, sticky=tk.W)
    
    def count_abnormal_values(self, vital):
        """Count the number of abnormal values for a vital sign"""
        count = 0
        for _, row in self.df.iterrows():
            value = row[vital]
            gender = row['Gender']
            if not self.is_value_normal(vital, value, gender):
                count += 1
        return count
    
    def update_demographics_analysis(self):
        """Update the demographics analysis charts"""
        # Update age distribution chart
        self.update_age_demographics_chart()
        
        # Update gender distribution chart
        self.update_gender_demographics_chart()
    
    def update_age_demographics_chart(self):
        """Update the age distribution chart in demographics tab"""
        # Clear existing chart
        for widget in self.age_demo_chart_frame.winfo_children():
            widget.destroy()
            
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create age bins
        age_bins = [0, 18, 30, 45, 60, 75, 90, 120]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-90', '90+']
        
        # Calculate age distribution
        age_counts = pd.cut(self.df['Age'], bins=age_bins).value_counts().sort_index()
        
        # Create bar chart
        bars = ax.bar(
            age_labels,
            age_counts.values,
            color=self.primary_color,
            alpha=0.7
        )
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                height + 0.1, 
                f'{height:.0f}', 
                ha='center', 
                va='bottom'
            )
        
        # Add labels and title
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Age Distribution')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.age_demo_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_gender_demographics_chart(self):
        """Update the gender distribution chart in demographics tab"""
        # Clear existing chart
        for widget in self.gender_chart_frame.winfo_children():
            widget.destroy()
            
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Calculate gender distribution
        gender_counts = self.df['Gender'].value_counts()
        
        # Create pie chart
        ax.pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct='%1.1f%%',
            colors=[self.primary_color, self.secondary_color, self.accent_color],
            startangle=90,
            shadow=True
        )
        
        # Add title
        ax.set_title('Gender Distribution')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.gender_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_correlation_analysis(self):
        """Update the correlation analysis chart"""
        # Clear existing chart
        for widget in self.correlation_chart_frame.winfo_children():
            widget.destroy()
            
        # Get numeric columns only
        numeric_cols = [col for col in self.df.columns 
                     if col not in ["Name", "Gender", "Disease_Prediction"]
                     and pd.api.types.is_numeric_dtype(self.df[col])]
        
        if not numeric_cols:
            # No numeric data to display
            ttk.Label(
                self.correlation_chart_frame,
                text="No numeric data to analyze",
                font=('Arial', 12)
            ).pack(expand=True)
            return
            
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
            fmt=".2f"
        )
        
        # Add title
        ax.set_title('Correlation Matrix of Health Metrics')
        
        # Embed chart in frame
        canvas = FigureCanvasTkAgg(fig, master=self.correlation_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_diet_view(self):
        """Create the diet recommendation view"""
        frame = ttk.Frame(self.content_frame, style='TFrame')
        
        # Title
        ttk.Label(
            frame, 
            text="Diet Recommendations", 
            style='Title.TLabel'
        ).pack(pady=(0, 20), anchor=tk.W)
        
        # Patient selection section
        selection_frame = ttk.Frame(frame, style='TFrame')
        selection_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(selection_frame, text="Select Patient:", style='TLabel').pack(side=tk.LEFT, padx=(0, 5))
        
        self.diet_patient_var = tk.StringVar()
        self.diet_patient_dropdown = ttk.Combobox(
            selection_frame,
            textvariable=self.diet_patient_var,
            state="readonly",
            width=30
        )
        self.diet_patient_dropdown.pack(side=tk.LEFT, padx=5)
        self.diet_patient_dropdown.bind("<<ComboboxSelected>>", self.show_patient_diet)
        
        ttk.Button(
            selection_frame,
            text="Show Recommendations",
            command=self.show_patient_diet
        ).pack(side=tk.LEFT, padx=5)
        
        # Health conditions summary
        conditions_frame = ttk.LabelFrame(frame, text="Health Conditions", style='TFrame')
        conditions_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.conditions_text = scrolledtext.ScrolledText(
            conditions_frame, 
            height=4, 
            wrap=tk.WORD, 
            font=('Arial', 10)
        )
        self.conditions_text.pack(fill=tk.X, padx=10, pady=10)
        self.conditions_text.config(state=tk.DISABLED)
        
        # Diet recommendations
        diet_frame = ttk.LabelFrame(frame, text="Personalized Diet Plan", style='TFrame')
        diet_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        # Notebook for diet sections
        self.diet_notebook = ttk.Notebook(diet_frame)
        self.diet_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Overview
        self.diet_overview_tab = ttk.Frame(self.diet_notebook, style='TFrame')
        self.diet_notebook.add(self.diet_overview_tab, text="Overview")
        
        # Tab 2: Foods to Eat
        self.foods_to_eat_tab = ttk.Frame(self.diet_notebook, style='TFrame')
        self.diet_notebook.add(self.foods_to_eat_tab, text="Foods to Eat")
        
        # Tab 3: Foods to Avoid
        self.foods_to_avoid_tab = ttk.Frame(self.diet_notebook, style='TFrame')
        self.diet_notebook.add(self.foods_to_avoid_tab, text="Foods to Avoid")
        
        # Tab 4: Meal Plan
        self.meal_plan_tab = ttk.Frame(self.diet_notebook, style='TFrame')
        self.diet_notebook.add(self.meal_plan_tab, text="Sample Meal Plan")
        
        # Setup each tab with content placeholders
        self.setup_diet_tabs()
        
        return frame
    
    def setup_diet_tabs(self):
        """Setup the diet recommendation tabs with placeholder content"""
        # Overview tab
        self.diet_overview_text = scrolledtext.ScrolledText(
            self.diet_overview_tab, 
            wrap=tk.WORD, 
            font=('Arial', 11)
        )
        self.diet_overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.diet_overview_text.config(state=tk.DISABLED)
        
        # Foods to Eat tab
        self.foods_to_eat_text = scrolledtext.ScrolledText(
            self.foods_to_eat_tab, 
            wrap=tk.WORD, 
            font=('Arial', 11)
        )
        self.foods_to_eat_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.foods_to_eat_text.config(state=tk.DISABLED)
        
        # Foods to Avoid tab
        self.foods_to_avoid_text = scrolledtext.ScrolledText(
            self.foods_to_avoid_tab, 
            wrap=tk.WORD, 
            font=('Arial', 11)
        )
        self.foods_to_avoid_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.foods_to_avoid_text.config(state=tk.DISABLED)
        
        # Meal Plan tab
        self.meal_plan_text = scrolledtext.ScrolledText(
            self.meal_plan_tab, 
            wrap=tk.WORD, 
            font=('Arial', 11)
        )
        self.meal_plan_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.meal_plan_text.config(state=tk.DISABLED)
    
    def update_diet_patient_dropdown(self):
        """Update the patient dropdown in the diet view"""
        if self.df is None:
            return
            
        # Get all patient names
        patient_options = self.df['Name'].tolist()
        
        # Update dropdown values
        self.diet_patient_dropdown['values'] = patient_options
        
        # Select first patient if available
        if patient_options:
            self.diet_patient_var.set(patient_options[0])
    
    def update_diet_recommendations(self):
        """Update the diet recommendations when the view is shown"""
        if self.df is None or len(self.df) == 0:
            return
            
        # If a patient is selected, show their recommendations
        if self.diet_patient_var.get():
            self.show_patient_diet()
    
    
    def show_patient_diet(self, event=None):
        """Show diet recommendations for the selected patient"""
        patient_name = self.diet_patient_var.get()
    
        if not patient_name:
            messagebox.showinfo("Selection Required", "Please select a patient from the dropdown.")
            return
        
        # Find patient in dataframe
        patient_df = self.df[self.df['Name'] == patient_name]
    
        if patient_df.empty:
            messagebox.showerror("Error", "Selected patient not found in database.")
            return
        
        # Get patient's first record (in case there are duplicates)
        patient = patient_df.iloc[0]
    
        # Get patient's conditions
        conditions = patient['Disease_Prediction'].split(", ") if patient['Disease_Prediction'] != "None" else []
    
        # Update conditions text
        self.conditions_text.config(state=tk.NORMAL)
        self.conditions_text.delete(1.0, tk.END)
    
        if not conditions:
            self.conditions_text.insert(tk.END, "No health conditions detected. The patient appears to be healthy.")
        else:
            self.conditions_text.insert(tk.END, f"The patient has the following health conditions:\n\n")
            for condition in conditions:
                self.conditions_text.insert(tk.END, f"• {condition}\n")
    
        self.conditions_text.config(state=tk.DISABLED)
    
        # Generate diet recommendations
        diet_recommendations = self.generate_diet_recommendations(conditions, patient)
    
        # Make sure diet_recommendations is not None before trying to access its keys
        if diet_recommendations is None:
            messagebox.showerror("Error", "Failed to generate diet recommendations.")
            return
        
        # Update overview tab
        self.diet_overview_text.config(state=tk.NORMAL)
        self.diet_overview_text.delete(1.0, tk.END)
        self.diet_overview_text.insert(tk.END, diet_recommendations.get('overview', 'No overview available.'))
        self.diet_overview_text.config(state=tk.DISABLED)
    
        # Update foods to eat tab
        self.foods_to_eat_text.config(state=tk.NORMAL)
        self.foods_to_eat_text.delete(1.0, tk.END)
        self.foods_to_eat_text.insert(tk.END, diet_recommendations.get('foods_to_eat', 'No food recommendations available.'))
        self.foods_to_eat_text.config(state=tk.DISABLED)
    
        # Update foods to avoid tab
        self.foods_to_avoid_text.config(state=tk.NORMAL)
        self.foods_to_avoid_text.delete(1.0, tk.END)
        self.foods_to_avoid_text.insert(tk.END, diet_recommendations.get('foods_to_avoid', 'No food restrictions available.'))
        self.foods_to_avoid_text.config(state=tk.DISABLED)
    
        # Update meal plan tab
        self.meal_plan_text.config(state=tk.NORMAL)
        self.meal_plan_text.delete(1.0, tk.END)
        self.meal_plan_text.insert(tk.END, diet_recommendations.get('meal_plan', 'No meal plan available.'))
        self.meal_plan_text.config(state=tk.DISABLED)

    def generate_diet_recommendations(self, conditions, patient):
        """Generate diet recommendations based on health conditions"""
        # Initialize with empty default values
        recommendations = {
            'overview': 'No specific dietary recommendations available.',
            'foods_to_eat': 'No specific foods to eat recommendations available.',
            'foods_to_avoid': 'No specific foods to avoid recommendations available.',
            'meal_plan': 'No specific meal plan available.'
        }
        
        try:
            # Default overview for healthy patient
            if not conditions:
                recommendations['overview'] = """Healthy Diet Recommendations

Based on your health assessment, you appear to be in good health with no specific conditions detected. 
A balanced diet is recommended to maintain your current health status and prevent future issues.

The following general dietary guidelines will help you maintain optimal health:
• Eat a variety of fruits and vegetables daily (aim for 5+ servings)
• Choose whole grains over refined carbohydrates
• Include lean proteins in most meals
• Consume healthy fats like those found in olive oil, avocados, and nuts
• Stay hydrated by drinking plenty of water throughout the day
• Limit added sugars, salt, and highly processed foods

This balanced approach to nutrition provides essential nutrients while helping maintain a healthy weight and reducing the risk of chronic diseases.
"""
                
                recommendations['foods_to_eat'] = """Recommended Foods for Overall Health

FRUITS & VEGETABLES:
• All fresh fruits - especially berries, apples, citrus fruits
• Dark leafy greens - spinach, kale, collard greens, arugula
• Cruciferous vegetables - broccoli, cauliflower, Brussels sprouts
• Colorful vegetables - bell peppers, carrots, tomatoes, sweet potatoes

PROTEINS:
• Lean meats - chicken breast, turkey, lean cuts of beef
• Fish - especially fatty fish like salmon, mackerel, and sardines (2-3 times per week)
• Plant proteins - beans, lentils, chickpeas, tofu, tempeh
• Eggs - preferably free-range

WHOLE GRAINS:
• Oats
• Brown rice
• Quinoa
• Whole wheat bread and pasta
• Barley, farro, buckwheat

HEALTHY FATS:
• Avocados
• Nuts and seeds - almonds, walnuts, flaxseeds, chia seeds
• Olive oil
• Fatty fish

DAIRY & ALTERNATIVES:
• Low-fat or fat-free milk
• Plain yogurt (especially Greek yogurt)
• Small amounts of cheese
• Fortified plant milks (almond, soy, oat)

BEVERAGES:
• Water (primary beverage)
• Green tea
• Black coffee (in moderation)
• Herbal teas
"""

                recommendations['foods_to_avoid'] = """Foods to Limit for Overall Health

While a healthy diet focuses on what to include rather than strict restrictions, these foods are best consumed only occasionally:

LIMIT THESE FOODS:
• Highly processed foods with long ingredient lists
• Fast food and fried foods
• Processed meats (bacon, sausage, hot dogs, deli meats)
• Sugary drinks (soda, sweetened juices, sports drinks)
• Refined carbohydrates (white bread, pastries, white rice)
• Excessive alcohol
• Foods with added sugars
• Foods high in sodium
• Foods with partially hydrogenated oils (trans fats)

HEALTHIER SUBSTITUTIONS:
• Instead of soda → Try water with lemon or sparkling water
• Instead of chips → Try nuts or air-popped popcorn
• Instead of white bread → Try whole grain bread
• Instead of ice cream → Try Greek yogurt with fruit
• Instead of candy → Try fresh or dried fruit
• Instead of processed meats → Try fresh-cooked lean meats
"""

                recommendations['meal_plan'] = """Sample Meal Plan for Overall Health

BREAKFAST OPTIONS:
• Oatmeal topped with berries, nuts, and a drizzle of honey
• Greek yogurt parfait with fresh fruit and granola
• Whole grain toast with avocado and a poached egg
• Vegetable omelet with side of fruit

LUNCH OPTIONS:
• Quinoa bowl with roasted vegetables, chickpeas, and tahini dressing
• Large salad with mixed greens, grilled chicken, vegetables, and olive oil dressing
• Whole grain wrap with hummus, turkey, and plenty of vegetables
• Lentil soup with side salad and whole grain roll

DINNER OPTIONS:
• Baked salmon with roasted sweet potatoes and steamed broccoli
• Stir-fry with tofu, plenty of vegetables, and brown rice
• Grilled chicken with quinoa and roasted vegetables
• Bean and vegetable chili with small side of corn bread

SNACK OPTIONS:
• Apple with 1-2 tablespoons of nut butter
• Small handful of mixed nuts and dried fruit
• Greek yogurt with berries
• Hummus with vegetables for dipping
• Whole grain crackers with avocado

HYDRATION:
• Aim for 8 glasses of water throughout the day
• Herbal teas are a good option, especially in colder weather
"""
                return recommendations
            
            # Build overview based on conditions
            overview = f"""Personalized Diet Recommendations

Based on your health assessment, a customized diet plan has been created to address your specific health conditions:
{', '.join(conditions)}

The dietary recommendations aim to:
• Help manage your current health conditions
• Provide necessary nutrients for your body
• Support your overall wellbeing
• Potentially reduce symptoms associated with your conditions

Following these recommendations may help improve your markers over time, but they should be used in conjunction with any treatments prescribed by your healthcare provider.

Please consult with your healthcare provider or a registered dietitian before making significant changes to your diet.
"""
            
            # Add condition-specific diet advice
            if "Anemia" in conditions:
                overview += "\nFor Anemia: Focus on iron-rich foods, vitamin C to improve iron absorption, and vitamin B12 sources.\n"
            
            if "Hypertension" in conditions:
                overview += "\nFor Hypertension: Follow a reduced-sodium diet with plenty of potassium-rich foods. The DASH diet approach is recommended.\n"
            
            if "Diabetes" in conditions:
                overview += "\nFor Diabetes: Focus on low glycemic index foods, consistent carbohydrate intake, and regular meal timing.\n"
            
            if "Heart Disease" in conditions:
                overview += "\nFor Heart Disease: Follow a heart-healthy diet low in saturated fats, with emphasis on omega-3 fatty acids, fiber, and plant sterols.\n"
            
            if "Vitamin D Deficiency" in conditions:
                overview += "\nFor Vitamin D Deficiency: Include vitamin D fortified foods and fatty fish in your diet. Consider supplementation as recommended by your doctor.\n"
            
            if "Kidney Disease" in conditions:
                overview += "\nFor Kidney Disease: Control protein, phosphorus, potassium, and sodium intake. Fluid restrictions may be necessary.\n"
            
            if "High Cholesterol" in conditions:
                overview += "\nFor High Cholesterol: Focus on heart-healthy fats, soluble fiber, and plant sterols. Limit saturated and trans fats.\n"
                
            recommendations['overview'] = overview
            
            # Foods to eat recommendations
            foods_to_eat = "RECOMMENDED FOODS\n\n"
            
            # Common healthy foods for everyone
            foods_to_eat += "GENERAL HEALTHY FOODS (RECOMMENDED FOR EVERYONE):\n"
            foods_to_eat += "• Fresh fruits and vegetables - aim for a variety of colors\n"
            foods_to_eat += "• Whole grains - brown rice, quinoa, whole wheat bread\n"
            foods_to_eat += "• Lean proteins - skinless poultry, fish, legumes\n"
            foods_to_eat += "• Healthy fats - olive oil, avocados, nuts\n\n"
            
            # Specific recommendations based on conditions
            if "Anemia" in conditions:
                foods_to_eat += "FOR ANEMIA:\n"
                foods_to_eat += "• Iron-rich foods: lean red meat, liver, shellfish, beans, spinach, fortified cereals\n"
                foods_to_eat += "• Vitamin C foods to improve iron absorption: citrus fruits, bell peppers, strawberries, tomatoes\n"
                foods_to_eat += "• Vitamin B12 sources: meat, fish, eggs, dairy, fortified plant milks\n"
                foods_to_eat += "• Folate-rich foods: dark leafy greens, legumes, avocados\n\n"
                
            if "Hypertension" in conditions:
                foods_to_eat += "FOR HYPERTENSION (DASH DIET):\n"
                foods_to_eat += "• Potassium-rich foods: bananas, potatoes, spinach, beans, yogurt\n"
                foods_to_eat += "• Calcium sources: low-fat dairy, fortified plant milks, leafy greens\n"
                foods_to_eat += "• Magnesium sources: nuts, seeds, whole grains, leafy greens\n"
                foods_to_eat += "• Fiber-rich foods: fruits, vegetables, whole grains, legumes\n\n"
                
            if "Diabetes" in conditions:
                foods_to_eat += "FOR DIABETES:\n"
                foods_to_eat += "• Low glycemic index foods: most non-starchy vegetables, most fruits, legumes, whole grains\n"
                foods_to_eat += "• High fiber foods: beans, lentils, oats, vegetables\n"
                foods_to_eat += "• Healthy fats: nuts, seeds, avocados, olive oil\n"
                foods_to_eat += "• Quality proteins: fish, skinless poultry, tofu, eggs\n\n"
                
            if "Heart Disease" in conditions:
                foods_to_eat += "FOR HEART DISEASE:\n"
                foods_to_eat += "• Omega-3 rich foods: fatty fish (salmon, mackerel, sardines), walnuts, flaxseeds\n"
                foods_to_eat += "• Foods with plant sterols: vegetables, fruits, whole grains, nuts\n"
                foods_to_eat += "• Soluble fiber: oats, barley, beans, fruits\n"
                foods_to_eat += "• Antioxidant-rich foods: berries, dark chocolate (70%+ cocoa), colorful vegetables\n\n"
                
            if "Vitamin D Deficiency" in conditions:
                foods_to_eat += "FOR VITAMIN D DEFICIENCY:\n"
                foods_to_eat += "• Vitamin D rich foods: fatty fish, egg yolks, mushrooms exposed to UV light\n"
                foods_to_eat += "• Vitamin D fortified foods: milk, plant milks, orange juice, cereals\n\n"
                
            if "Kidney Disease" in conditions:
                foods_to_eat += "FOR KIDNEY DISEASE:\n"
                foods_to_eat += "• Lower protein options: egg whites, smaller portions of meat and fish\n"
                foods_to_eat += "• Lower phosphorus foods: rice milk (unfortified), breads without whole grains, corn or rice cereals\n"
                foods_to_eat += "• Lower potassium fruits: apples, berries, grapes, pineapple\n"
                foods_to_eat += "• Lower potassium vegetables: carrots, green beans, cabbage, lettuce\n\n"
                
            if "High Cholesterol" in conditions:
                foods_to_eat += "FOR HIGH CHOLESTEROL:\n"
                foods_to_eat += "• Soluble fiber: oats, barley, fruits, legumes\n"
                foods_to_eat += "• Omega-3 fatty acids: fatty fish, flaxseeds, chia seeds\n"
                foods_to_eat += "• Plant sterols: vegetables, fruits, nuts, seeds\n"
                foods_to_eat += "• Soy protein: tofu, tempeh, edamame, soy milk\n\n"
                
            recommendations['foods_to_eat'] = foods_to_eat
            
            # Foods to avoid recommendations
            foods_to_avoid = "FOODS TO LIMIT OR AVOID\n\n"
            
            # Common foods to avoid for everyone
            foods_to_avoid += "GENERAL FOODS TO LIMIT (FOR EVERYONE):\n"
            foods_to_avoid += "• Highly processed foods with artificial additives\n"
            foods_to_avoid += "• Excessive sugar and sweetened beverages\n"
            foods_to_avoid += "• Trans fats and fried foods\n\n"
            
            # Specific recommendations based on conditions
            if "Anemia" in conditions:
                foods_to_avoid += "FOR ANEMIA:\n"
                foods_to_avoid += "• Foods that inhibit iron absorption when eaten with iron-rich foods: coffee, tea, excessive calcium\n"
                foods_to_avoid += "• Foods with phytates if consumed with iron sources: whole grains, legumes (spacing these out from iron-rich meals can help)\n\n"
                
            if "Hypertension" in conditions:
                foods_to_avoid += "FOR HYPERTENSION:\n"
                foods_to_avoid += "• High sodium foods: processed meats, canned soups, frozen dinners, salty snacks\n"
                foods_to_avoid += "• Alcohol (limit to moderate consumption if at all)\n"
                foods_to_avoid += "• Caffeine (monitor its effects on your blood pressure)\n\n"
                
            if "Diabetes" in conditions:
                foods_to_avoid += "FOR DIABETES:\n"
                foods_to_avoid += "• High glycemic index foods: white bread, white rice, candy, sugary drinks\n"
                foods_to_avoid += "• Added sugars: desserts, sweetened beverages, many packaged foods\n"
                foods_to_avoid += "• Large portions of fruit juice or dried fruit\n"
                foods_to_avoid += "• Excessive carbohydrates in one sitting\n\n"
                
            if "Heart Disease" in conditions:
                foods_to_avoid += "FOR HEART DISEASE:\n"
                foods_to_avoid += "• Foods high in saturated fats: fatty meats, full-fat dairy, coconut oil\n"
                foods_to_avoid += "• Trans fats: fried foods, baked goods with partially hydrogenated oils\n"
                foods_to_avoid += "• High sodium foods: processed foods, canned soups, salty snacks\n"
                foods_to_avoid += "• Excessive sugar: desserts, sweetened beverages\n\n"
                
            if "Vitamin D Deficiency" in conditions:
                foods_to_avoid += "FOR VITAMIN D DEFICIENCY:\n"
                foods_to_avoid += "• No specific foods to avoid, but be aware that few foods naturally contain vitamin D\n"
                foods_to_avoid += "• Consuming extremely low fat diets may reduce vitamin D absorption\n\n"
                
            if "Kidney Disease" in conditions:
                foods_to_avoid += "FOR KIDNEY DISEASE:\n"
                foods_to_avoid += "• High phosphorus foods: dairy, nuts, seeds, whole grains, processed foods with phosphate additives\n"
                foods_to_avoid += "• High potassium foods: bananas, oranges, potatoes, tomatoes, avocados\n"
                foods_to_avoid += "• High sodium foods: processed foods, canned foods, salty snacks\n"
                foods_to_avoid += "• Excessive protein: large portions of meat, poultry, fish\n\n"
                
            if "High Cholesterol" in conditions:
                foods_to_avoid += "FOR HIGH CHOLESTEROL:\n"
                foods_to_avoid += "• Foods high in saturated fats: fatty meats, full-fat dairy, butter, coconut oil\n"
                foods_to_avoid += "• Trans fats: fried foods, commercial baked goods, anything with partially hydrogenated oils\n"
                foods_to_avoid += "• Excessive dietary cholesterol (less important than saturated fat): organ meats, egg yolks in large quantities\n\n"
                
            recommendations['foods_to_avoid'] = foods_to_avoid
            
            # Sample meal plan based on conditions
            if "Anemia" in conditions or "Iron Deficiency" in conditions:
                breakfast = "BREAKFAST:\n"
                breakfast += "• Iron-fortified cereal with strawberries (vitamin C to enhance iron absorption)\n"
                breakfast += "• 1 hard-boiled egg\n"
                breakfast += "• Glass of orange juice (vitamin C)\n\n"
                
                lunch = "LUNCH:\n"
                lunch += "• Spinach salad with grilled chicken, bell peppers, and citrus dressing\n"
                lunch += "• Whole grain roll\n"
                lunch += "• Water with lemon\n\n"
                
                dinner = "DINNER:\n"
                dinner += "• Lean beef stir-fry with broccoli, bell peppers, and brown rice\n"
                dinner += "• Strawberry and spinach side salad\n"
                dinner += "• Water\n\n"
                
                snacks = "SNACKS:\n"
                snacks += "• Handful of dried apricots and pumpkin seeds\n"
                snacks += "• Hummus with bell pepper strips\n\n"
                
            elif "Hypertension" in conditions:
                breakfast = "BREAKFAST:\n"
                breakfast += "• Overnight oats with banana and a sprinkle of cinnamon\n"
                breakfast += "• Low-fat yogurt\n"
                breakfast += "• Herbal tea\n\n"
                
                lunch = "LUNCH:\n"
                lunch += "• Quinoa bowl with grilled chicken, roasted vegetables, and avocado\n"
                lunch += "• Mixed berries\n"
                lunch += "• Water\n\n"
                
                dinner = "DINNER:\n"
                dinner += "• Baked salmon with dill\n"
                dinner += "• Steamed spinach and sweet potato\n"
                dinner += "• Mixed green salad with olive oil and lemon dressing\n"
                dinner += "• Water with lime\n\n"
                
                snacks = "SNACKS:\n"
                snacks += "• Unsalted mixed nuts\n"
                snacks += "• Apple slices\n\n"
                
            elif "Diabetes" in conditions:
                breakfast = "BREAKFAST:\n"
                breakfast += "• Vegetable omelet (2 eggs with spinach, tomatoes, mushrooms)\n"
                breakfast += "• 1 slice whole grain toast\n"
                breakfast += "• 1/4 avocado\n"
                breakfast += "• Unsweetened tea or coffee\n\n"
                
                lunch = "LUNCH:\n"
                lunch += "• Grilled chicken salad with mixed greens, cucumber, cherry tomatoes\n"
                lunch += "• 1/2 cup quinoa\n"
                lunch += "• Olive oil and vinegar dressing\n"
                lunch += "• Water\n\n"
                
                dinner = "DINNER:\n"
                dinner += "• Baked fish with herbs\n"
                dinner += "• 1/2 cup lentils\n"
                dinner += "• Roasted non-starchy vegetables (broccoli, cauliflower, bell peppers)\n"
                dinner += "• Water\n\n"
                
                snacks = "SNACKS:\n"
                snacks += "• Small apple with 1 tablespoon almond butter\n"
                snacks += "• 1/4 cup cottage cheese with cucumber slices\n\n"
                
            elif "Heart Disease" in conditions or "High Cholesterol" in conditions:
                breakfast = "BREAKFAST:\n"
                breakfast += "• Oatmeal with ground flaxseeds, berries, and cinnamon\n"
                breakfast += "• Unsweetened plant milk\n"
                breakfast += "• Green tea\n\n"
                
                lunch = "LUNCH:\n"
                lunch += "• Bean and vegetable soup\n"
                lunch += "• Side salad with olive oil and lemon dressing\n"
                lunch += "• 1 piece of fruit\n"
                lunch += "• Water\n\n"
                
                dinner = "DINNER:\n"
                dinner += "• Grilled salmon with herbs\n"
                dinner += "• Steamed vegetables (broccoli, carrots)\n"
                dinner += "• 1/2 cup brown rice\n"
                dinner += "• Water\n\n"
                
                snacks = "SNACKS:\n"
                snacks += "• Small handful of walnuts\n"
                snacks += "• Apple slices\n\n"
                
            elif "Kidney Disease" in conditions:
                breakfast = "BREAKFAST:\n"
                breakfast += "• Egg whites (2-3) with low-potassium vegetables (bell peppers, onions)\n"
                breakfast += "• 1 slice white toast with small amount of unsalted butter\n"
                breakfast += "• 1/2 cup berries\n"
                breakfast += "• Water\n\n"
                
                lunch = "LUNCH:\n"
                lunch += "• Chicken sandwich on white bread with lettuce and cucumber\n"
                lunch += "• Low-potassium fruit (apple or grapes)\n"
                lunch += "• Water\n\n"
                
                dinner = "DINNER:\n"
                dinner += "• Small portion of lean protein (chicken, fish)\n"
                dinner += "• White rice\n"
                dinner += "• Green beans or carrots\n"
                dinner += "• Water\n\n"
                
                snacks = "SNACKS:\n"
                snacks += "• Rice cakes with small amount of unsalted butter\n"
                snacks += "• Low-potassium fruit (pineapple)\n\n"
                
            else:
                # Default healthy meal plan
                breakfast = "BREAKFAST:\n"
                breakfast += "• Oatmeal topped with berries, nuts, and a drizzle of honey\n"
                breakfast += "• Greek yogurt on the side\n"
                breakfast += "• Green tea or water\n\n"
                
                lunch = "LUNCH:\n"
                lunch += "• Mediterranean quinoa bowl with chickpeas, cucumber, tomatoes, olives, feta\n"
                lunch += "• Olive oil and lemon dressing\n"
                lunch += "• 1 piece of fruit\n"
                lunch += "• Water\n\n"
                
                dinner = "DINNER:\n"
                dinner += "• Grilled salmon with herbs\n"
                dinner += "• Roasted vegetables (sweet potatoes, Brussels sprouts, carrots)\n"
                dinner += "• Quinoa or brown rice\n"
                dinner += "• Water\n\n"
                
                snacks = "SNACKS:\n"
                snacks += "• Apple with almond butter\n"
                snacks += "• Hummus with vegetable sticks\n"
                snacks += "• Small handful of mixed nuts\n\n"
                
            meal_plan = breakfast + lunch + dinner + snacks
            meal_plan += "NOTES:\n"
            meal_plan += "• Adjust portion sizes based on your specific caloric needs\n"
            meal_plan += "• Drink plenty of water throughout the day\n"
            meal_plan += "• This meal plan is a general guideline - consult with a dietitian for a fully personalized plan\n"
            
            recommendations['meal_plan'] = meal_plan
        
        except Exception as e:
            print(f"Error generating diet recommendations: {str(e)}")
            # Return the default recommendations dictionary even if an error occurs
            
        return recommendations
    
    def display_diet_recommendations(self, predicted_diseases):
        """Display diet recommendations for the predicted diseases"""
        if not hasattr(self, 'diet_recommendation_frame') or not self.diet_recommendation_frame:
            # If the frame doesn't exist, create it
            self.diet_recommendation_frame = ttk.LabelFrame(
                self.prediction_frame.master, 
                text="Diet Recommendations", 
                style='TFrame'
            )
    
        # Clear any existing diet recommendation frame
        if self.diet_recommendation_frame.winfo_manager():
            self.diet_recommendation_frame.pack_forget()
        
        # Pack the frame
        self.diet_recommendation_frame.pack(fill=tk.X, pady=(0, 15), padx=5, after=self.prediction_frame)
    
        # Clear existing content
        for widget in self.diet_recommendation_frame.winfo_children():
            widget.destroy()
    
        if not predicted_diseases:
            # No diseases, provide general recommendations
            recommendations_text = scrolledtext.ScrolledText(
                self.diet_recommendation_frame,
                height=10,
                wrap=tk.WORD,
                font=('Arial', 11)
            )
            recommendations_text.pack(fill=tk.BOTH, padx=15, pady=15)
        
            general_diet = """Healthy Diet Recommendations:

Since no specific health conditions were detected, here are general healthy eating guidelines:

- Eat a variety of fruits and vegetables daily (aim for 5+ servings)
- Choose whole grains over refined carbohydrates
- Include lean proteins in most meals 
- Consume healthy fats like olive oil, avocados, and nuts
- Limit added sugars, salt, and highly processed foods
- Stay hydrated by drinking plenty of water throughout the day

These recommendations support overall good health and help prevent chronic diseases.
"""
            recommendations_text.insert(tk.END, general_diet)
            recommendations_text.config(state=tk.DISABLED)
        else:
            # Get diet recommendations based on diseases
            patient_data = {
                'Disease_Prediction': ', '.join(predicted_diseases),
                'Age': self.age_var.get(),
                'Gender': self.gender_var.get()
            }
        
            diet_recommendations = self.generate_diet_recommendations(predicted_diseases, patient_data)
        
            # Create tabbed interface for diet recommendations
            diet_notebook = ttk.Notebook(self.diet_recommendation_frame)
            diet_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
            # Overview tab
            overview_tab = ttk.Frame(diet_notebook)
            diet_notebook.add(overview_tab, text="Overview")
        
            overview_text = scrolledtext.ScrolledText(
                overview_tab,
                height=8,
                wrap=tk.WORD,
                font=('Arial', 11)
            )
            overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            overview_text.insert(tk.END, diet_recommendations['overview'])
            overview_text.config(state=tk.DISABLED)
        
            # Foods to eat tab
            foods_tab = ttk.Frame(diet_notebook)
            diet_notebook.add(foods_tab, text="Foods to Eat")
        
            foods_text = scrolledtext.ScrolledText(
                foods_tab,
                height=8,
                wrap=tk.WORD,
                font=('Arial', 11)
            )
            foods_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            foods_text.insert(tk.END, diet_recommendations['foods_to_eat'])
            foods_text.config(state=tk.DISABLED)
        
            # Foods to avoid tab
            avoid_tab = ttk.Frame(diet_notebook)
            diet_notebook.add(avoid_tab, text="Foods to Avoid")
        
            avoid_text = scrolledtext.ScrolledText(
                avoid_tab,
                height=8,
                wrap=tk.WORD,
                font=('Arial', 11)
            )
            avoid_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            avoid_text.insert(tk.END, diet_recommendations['foods_to_avoid'])
            avoid_text.config(state=tk.DISABLED)
    
        
    def view_patient_details(self, patient):
        """Show detailed patient information in a popup window"""
        # Create a new window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Patient Details: {patient['Name']}")
        details_window.geometry("800x600")
        details_window.minsize(600, 400)
        
        # Apply styling
        details_window.configure(bg=self.bg_color)
        
        # Create a frame for all content
        main_frame = ttk.Frame(details_window, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Patient header info
        header_frame = ttk.Frame(main_frame, style='TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Patient name (large)
        ttk.Label(
            header_frame,
            text=patient['Name'],
            font=('Arial', 18, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W)
        
        # Basic demographics
        demo_text = f"Age: {patient['Age']} | Gender: {patient['Gender']}"
        ttk.Label(
            header_frame,
            text=demo_text,
            font=('Arial', 12),
            style='TLabel'
        ).pack(anchor=tk.W)
        
        # Health status
        if patient['Disease_Prediction'] == "None":
            status_frame = ttk.Frame(header_frame, style='TFrame')
            status_frame.pack(anchor=tk.W, pady=5)
            
            status_indicator = tk.Frame(status_frame, bg="green", width=15, height=15)
            status_indicator.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(
                status_frame,
                text="Healthy - No conditions detected",
                font=('Arial', 12),
                foreground="green",
                style='TLabel'
            ).pack(side=tk.LEFT)
        else:
            status_frame = ttk.Frame(header_frame, style='TFrame')
            status_frame.pack(anchor=tk.W, pady=5)
            
            status_indicator = tk.Frame(status_frame, bg=self.warning_color, width=15, height=15)
            status_indicator.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(
                status_frame,
                text=f"Conditions: {patient['Disease_Prediction']}",
                font=('Arial', 12),
                foreground=self.warning_color,
                style='TLabel'
            ).pack(side=tk.LEFT)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Vital Signs
        vitals_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(vitals_tab, text="Vital Signs")
        
        # Tab 2: Predictions
        predictions_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(predictions_tab, text="Disease Prediction")
        
        # Tab 3: Diet Recommendations
        diet_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(diet_tab, text="Diet Recommendations")
        
        # Populate Vital Signs tab
        self.populate_vitals_tab(vitals_tab, patient)
        
        # Populate Predictions tab
        self.populate_predictions_tab(predictions_tab, patient)
        
        # Populate Diet Recommendations tab
        self.populate_diet_tab(diet_tab, patient)
        
        # Buttons at bottom
        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Close button
        ttk.Button(
            button_frame,
            text="Close",
            command=details_window.destroy
        ).pack(side=tk.RIGHT)
        
        # Edit button
        ttk.Button(
            button_frame,
            text="Edit Patient",
            command=lambda: self.edit_patient(patient, details_window)
        ).pack(side=tk.RIGHT, padx=10)
        
        # Center the window on screen
        details_window.update_idletasks()
        width = details_window.winfo_width()
        height = details_window.winfo_height()
        x = (details_window.winfo_screenwidth() // 2) - (width // 2)
        y = (details_window.winfo_screenheight() // 2) - (height // 2)
        details_window.geometry(f'+{x}+{y}')
    
    def populate_vitals_tab(self, tab_frame, patient):
        """Populate the vitals tab with patient health metrics"""
        # Create scrollable frame
        canvas = tk.Canvas(tab_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a frame inside canvas
        vitals_frame = ttk.Frame(canvas, style='TFrame')
        canvas_window = canvas.create_window(
            (0, 0), window=vitals_frame, anchor="nw", width=canvas.winfo_width()
        )
        
        # Configure canvas scrolling behavior
        vitals_frame.bind(
            "<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", 
            lambda e: canvas.itemconfig(canvas_window, width=e.width)
        )
        
        # Get all vital sign columns
        vital_columns = [col for col in patient.index if col not in ["Name", "Age", "Gender", "Disease_Prediction"]]
        
        # Create a grid for vital signs (3 columns)
        for i in range(3):
            vitals_frame.columnconfigure(i, weight=1)
            
        row, col = 0, 0
        for vital in vital_columns:
            # Get value and normal range
            value = patient[vital]
            normal_range = self.get_normal_range(vital)
            
            # Determine if value is normal
            is_normal = self.is_value_normal(vital, value, patient['Gender'])
            
            # Create a card for this vital sign
            card = tk.Frame(
                vitals_frame,
                bg="white",
                highlightbackground="#e0e0e0",
                highlightthickness=1,
                padx=10,
                pady=10
            )
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Vital name
            tk.Label(
                card,
                text=vital.replace("_", " "),
                font=('Arial', 11, 'bold'),
                bg="white"
            ).pack(anchor=tk.W)
            
            # Value with color coding
            value_frame = tk.Frame(card, bg="white")
            value_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                value_frame,
                text=f"{value:.1f}",
                font=('Arial', 16),
                fg="green" if is_normal else self.warning_color,
                bg="white"
            ).pack(side=tk.LEFT)
            
            # Unit
            unit = self.get_unit(vital)
            if unit:
                tk.Label(
                    value_frame,
                    text=unit,
                    font=('Arial', 12),
                    fg="gray",
                    bg="white"
                ).pack(side=tk.LEFT, padx=(5, 0), anchor=tk.S)
            
            # Normal range info
            tk.Label(
                card,
                text=f"Normal range: {normal_range}",
                font=('Arial', 9),
                fg="gray",
                bg="white"
            ).pack(anchor=tk.W)
            
            # Status indicator
            status_text = "Normal" if is_normal else "Abnormal"
            status_color = "green" if is_normal else self.warning_color
            
            tk.Label(
                card,
                text=status_text,
                font=('Arial', 10),
                fg=status_color,
                bg="white"
            ).pack(anchor=tk.W, pady=(5, 0))
            
            # Update grid position
            col += 1
            if col >= 3:  # 3 columns per row
                col = 0
                row += 1
    
    def get_normal_range(self, vital):
        """Get the normal range for a vital sign"""
        ranges = {
            "Hemoglobin": "13.5-17.5 g/dL (M), 12.0-16.0 g/dL (F)",
            "BP_Systolic": "90-140 mmHg",
            "BP_Diastolic": "60-90 mmHg",
            "Heart_Rate": "60-100 bpm",
            "HbA1c": "<5.7%",
            "Vitamin_D": ">30 ng/mL",
            "LDL": "<100 mg/dL",
            "Iron": "60-170 μg/dL",
            "Creatinine": "0.7-1.3 mg/dL",
            "MCH": "27-33 pg",
            "MCHC": "33-36 g/dL",
            "CRP": "<3.0 mg/L"
        }
        
        return ranges.get(vital, "Not specified")
    
    def get_unit(self, vital):
        """Get the unit for a vital sign"""
        units = {
            "Hemoglobin": "g/dL",
            "BP_Systolic": "mmHg",
            "BP_Diastolic": "mmHg",
            "Heart_Rate": "bpm",
            "HbA1c": "%",
            "Vitamin_D": "ng/mL",
            "LDL": "mg/dL",
            "Iron": "μg/dL",
            "Creatinine": "mg/dL",
            "MCH": "pg",
            "MCHC": "g/dL",
            "CRP": "mg/L"
        }
        
        return units.get(vital, "")
    
    def is_value_normal(self, vital, value, gender):
        """Check if a vital sign value is within normal range"""
        # Define thresholds for normal values
        thresholds = {
            "Hemoglobin": lambda x, g: 13.5 <= x <= 17.5 if g == "Male" else 12.0 <= x <= 16.0,
            "BP_Systolic": lambda x, g: 90 <= x <= 140,
            "BP_Diastolic": lambda x, g: 60 <= x <= 90,
            "Heart_Rate": lambda x, g: 60 <= x <= 100,
            "HbA1c": lambda x, g: x < 5.7,
            "Vitamin_D": lambda x, g: x >= 30,
            "LDL": lambda x, g: x < 100,
            "Iron": lambda x, g: 60 <= x <= 170,
            "Creatinine": lambda x, g: 0.7 <= x <= 1.3,
            "MCH": lambda x, g: 27 <= x <= 33,
            "MCHC": lambda x, g: 33 <= x <= 36,
            "CRP": lambda x, g: x < 3.0
        }
        
        # If threshold defined, check if value is normal
        if vital in thresholds:
            return thresholds[vital](value, gender)
        
        # Default to normal if no threshold defined
        return True
    
    def populate_predictions_tab(self, tab_frame, patient):
        """Populate the predictions tab with disease predictions"""
        # If we have a model, make a new prediction
        if self.model is not None and self.scaler is not None and self.mlb is not None:
            # Create prediction section
            prediction_frame = ttk.Frame(tab_frame, style='TFrame')
            prediction_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
            ttk.Label(
                prediction_frame,
                text="Disease Prediction Results",
                font=('Arial', 14, 'bold'),
                style='TLabel'
            ).pack(anchor=tk.W, pady=(0, 20))
        
            try:
                # Get the exact feature columns the scaler expects (first 16)
                feature_cols = [col for col in self.df.columns if col not in ["Name", "Gender", "Disease_Prediction"]][:16]
                print(f"Patient details: Using first 16 feature columns: {feature_cols}")
            
                # Build feature vector explicitly using the first 16 columns
                features = []
                for col in feature_cols:
                    if col in patient:
                        features.append(patient[col])
                    else:
                        # Use a reasonable default if column is missing
                        features.append(0.0)
                        print(f"Missing column in patient data: {col}")
            
                # Convert to numpy array and reshape
                features = np.array(features).reshape(1, -1)
                print(f"Patient details: Feature array shape: {features.shape}")
            
                # Scale features
                scaled_features = self.scaler.transform(features)
            
                # Make prediction
                prediction = self.model.predict(scaled_features)
            
                # Create chart for disease probabilities
                chart_frame = ttk.Frame(prediction_frame, style='TFrame')
                chart_frame.pack(fill=tk.BOTH, expand=True)
            
                # Get predicted diseases
                predicted_diseases = []
                for i, val in enumerate(prediction[0]):
                    if val == 1 and i < len(self.mlb.classes_):
                        predicted_diseases.append(self.mlb.classes_[i])
            
                # Get probabilities for each disease
                disease_probs = []
                for i, disease in enumerate(self.mlb.classes_):
                    # Get estimator for this disease
                    if hasattr(self.model, 'estimators_') and i < len(self.model.estimators_):
                        estimator = self.model.estimators_[i]
                        if hasattr(estimator, 'predict_proba'):
                            # Get probability of the positive class
                            prob = estimator.predict_proba(scaled_features)[0][1]
                            disease_probs.append((disease, prob))
            
                # Sort by probability
                disease_probs.sort(key=lambda x: x[1], reverse=True)
            
                print(f"Patient details - Predicted diseases: {predicted_diseases}")
                print(f"Patient details - Top disease probabilities: {disease_probs[:5]}")
            
                # Create figure for chart
                fig, ax = plt.subplots(figsize=(8, 5))
            
                # Plot horizontal bar chart
                diseases = [d[0] for d in disease_probs[:10]]  # Top 10 diseases
                probs = [d[1] * 100 for d in disease_probs[:10]]  # Convert to percentage
            
                bars = ax.barh(diseases, probs, color=self.primary_color)
            
                # Add percentage labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(
                        width + 1, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', 
                        ha='left', 
                        va='center'
                    )
            
                # Add labels and title
                ax.set_xlabel('Probability (%)')
                ax.set_title('Disease Probability')
                ax.set_xlim(0, 100)
            
                # Embed chart
                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
                # High risk diseases (>30% probability)
                high_risk = [(d, p) for d, p in disease_probs if p > 0.3]
            
                if high_risk:
                    risk_frame = ttk.Frame(prediction_frame, style='TFrame')
                    risk_frame.pack(fill=tk.X, pady=(20, 0))
                
                    ttk.Label(
                        risk_frame,
                        text="High Risk Conditions:",
                        font=('Arial', 12, 'bold'),
                        foreground=self.warning_color,
                        style='TLabel'
                    ).pack(anchor=tk.W)
                
                    for disease, prob in high_risk:
                        ttk.Label(
                            risk_frame,
                            text=f"• {disease} ({prob*100:.1f}% probability)",
                            font=('Arial', 11),
                            foreground=self.warning_color,
                            style='TLabel'
                        ).pack(anchor=tk.W, padx=(20, 0))
                else:
                    ttk.Label(
                        prediction_frame,
                        text="No high-risk conditions detected.",
                        font=('Arial', 12),
                        foreground="green",
                        style='TLabel'
                    ).pack(anchor=tk.W, pady=(20, 0))
                
            except Exception as e:
                # Error making prediction
                import traceback
                error_details = traceback.format_exc()
                error_message = f"Error making prediction: {str(e)}"
                print(f"Patient details prediction error: {error_message}\n{error_details}")
            
                ttk.Label(
                    prediction_frame,
                    text=error_message,
                    font=('Arial', 12),
                    foreground=self.warning_color,
                    style='TLabel'
                ).pack(anchor=tk.W)
        else:
            # No model available
            ttk.Label(
                tab_frame,
                text="Prediction model not available. Cannot make predictions.",
                font=('Arial', 12),
                style='TLabel'
            ).pack(anchor=tk.CENTER, expand=True)

    def populate_diet_tab(self, tab_frame, patient):
        """Populate the diet recommendations tab"""
        # Get patient's conditions
        conditions = patient['Disease_Prediction'].split(", ") if patient['Disease_Prediction'] != "None" else []
        
        # Diet recommendations frame
        diet_frame = ttk.Frame(tab_frame, style='TFrame')
        diet_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        ttk.Label(
            diet_frame,
            text="Personalized Diet Recommendations",
            font=('Arial', 14, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(0, 20))
        
        # Generate diet recommendations based on conditions
        diet_recommendations = self.generate_diet_recommendations(conditions, patient)
        
        # Display diet overview
        ttk.Label(
            diet_frame,
            text="Diet Overview",
            font=('Arial', 12, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Overview text
        overview_text = tk.Text(
            diet_frame,
            wrap=tk.WORD,
            height=6,
            font=('Arial', 11),
            padx=10,
            pady=10
        )
        overview_text.pack(fill=tk.X)
        overview_text.insert(tk.END, diet_recommendations['overview'])
        overview_text.config(state=tk.DISABLED)
        
        # Foods to eat
        ttk.Label(
            diet_frame,
            text="Recommended Foods",
            font=('Arial', 12, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(20, 10))
        
        foods_text = tk.Text(
            diet_frame,
            wrap=tk.WORD,
            height=6,
            font=('Arial', 11),
            padx=10,
            pady=10
        )
        foods_text.pack(fill=tk.X)
        foods_text.insert(tk.END, diet_recommendations['foods_to_eat'])
        foods_text.config(state=tk.DISABLED)
        
        # Foods to avoid
        ttk.Label(
            diet_frame,
            text="Foods to Avoid",
            font=('Arial', 12, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(20, 10))
        
        avoid_text = tk.Text(
            diet_frame,
            wrap=tk.WORD,
            height=6,
            font=('Arial', 11),
            padx=10,
            pady=10
        )
        avoid_text.pack(fill=tk.X)
        avoid_text.insert(tk.END, diet_recommendations['foods_to_avoid'])
        avoid_text.config(state=tk.DISABLED)
    
    def edit_patient(self, patient, parent_window):
        """Edit an existing patient's data"""
        # Create edit window
        edit_window = tk.Toplevel(parent_window)
        edit_window.title(f"Edit Patient: {patient['Name']}")
        edit_window.geometry("600x700")
        edit_window.minsize(500, 600)
        
        # Apply styling
        edit_window.configure(bg=self.bg_color)
        
        # Create a scrollable frame
        canvas = tk.Canvas(edit_window, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(edit_window, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a frame inside canvas
        edit_frame = ttk.Frame(canvas, style='TFrame')
        canvas_window = canvas.create_window(
            (0, 0), window=edit_frame, anchor="nw", width=canvas.winfo_width()
        )
        
        # Configure canvas scrolling behavior
        edit_frame.bind(
            "<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", 
            lambda e: canvas.itemconfig(canvas_window, width=e.width)
        )
        
        # Header
        ttk.Label(
            edit_frame,
            text=f"Edit Patient: {patient['Name']}",
            font=('Arial', 14, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, padx=20, pady=(20, 10))
        
        # Personal info section
        personal_frame = ttk.LabelFrame(edit_frame, text="Personal Information", style='TFrame')
        personal_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Form grid
        form_grid = ttk.Frame(personal_frame, style='TFrame')
        form_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Variables to store edited values
        edit_vars = {}
        
        # Name field
        ttk.Label(form_grid, text="Name:", style='TLabel').grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        edit_vars['Name'] = tk.StringVar(value=patient['Name'])
        ttk.Entry(form_grid, textvariable=edit_vars['Name'], width=30).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Age field
        ttk.Label(form_grid, text="Age:", style='TLabel').grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        edit_vars['Age'] = tk.StringVar(value=str(patient['Age']))
        ttk.Spinbox(form_grid, from_=0, to=120, textvariable=edit_vars['Age'], width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Gender field
        ttk.Label(form_grid, text="Gender:", style='TLabel').grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        edit_vars['Gender'] = tk.StringVar(value=patient['Gender'])
        ttk.Combobox(
            form_grid, 
            textvariable=edit_vars['Gender'],
            values=["Male", "Female", "Other"],
            state="readonly",
            width=10
        ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Vital signs section
        vitals_frame = ttk.LabelFrame(edit_frame, text="Vital Health Metrics", style='TFrame')
        vitals_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Vitals grid
        vitals_grid = ttk.Frame(vitals_frame, style='TFrame')
        vitals_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Configure grid columns
        for i in range(4):
            vitals_grid.columnconfigure(i, weight=1)
        
        # Add fields for all vital signs
        vital_cols = [col for col in patient.index if col not in ["Name", "Age", "Gender", "Disease_Prediction"]]
        
        for i, col in enumerate(vital_cols):
            row = i // 2
            col_offset = (i % 2) * 2
            
            # Label with unit
            unit = self.get_unit(col)
            label_text = f"{col.replace('_', ' ')}:"
            if unit:
                label_text = f"{col.replace('_', ' ')} ({unit}):"
                
            ttk.Label(vitals_grid, text=label_text, style='TLabel').grid(
                row=row, column=col_offset, sticky=tk.W, padx=5, pady=5
            )
            
            # Input field
            edit_vars[col] = tk.DoubleVar(value=float(patient[col]))
            ttk.Spinbox(
                vitals_grid,
                from_=0,
                to=1000,
                textvariable=edit_vars[col],
                width=10,
                increment=0.1
            ).grid(row=row, column=col_offset+1, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(edit_frame, style='TFrame')
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Save button
        ttk.Button(
            button_frame,
            text="Save Changes",
            style='Primary.TButton',
            command=lambda: self.save_patient_edits(patient, edit_vars, edit_window, parent_window)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Cancel button
        ttk.Button(
            button_frame,
            text="Cancel",
            command=edit_window.destroy
        ).pack(side=tk.LEFT)
    
    def save_patient_edits(self, original_patient, edit_vars, edit_window, parent_window):
        """Save the edited patient data"""
        try:
            # Validate inputs
            name = edit_vars['Name'].get().strip()
            if not name:
                messagebox.showerror("Validation Error", "Name cannot be empty")
                return
                
            try:
                age = int(edit_vars['Age'].get())
                if age < 0 or age > 120:
                    raise ValueError("Invalid age")
            except ValueError:
                messagebox.showerror("Validation Error", "Age must be a number between 0 and 120")
                return
            
            # Create updated patient record
            updated_patient = original_patient.copy()
            updated_patient['Name'] = name
            updated_patient['Age'] = age
            updated_patient['Gender'] = edit_vars['Gender'].get()
            
            # Update vital signs
            for col, var in edit_vars.items():
                if col not in ['Name', 'Age', 'Gender']:
                    updated_patient[col] = var.get()
            
            # Find the patient in the dataframe by index
            patient_idx = self.df.loc[self.df['Name'] == original_patient['Name']].index
            
            if len(patient_idx) > 0:
                # Update the dataframe
                for col in updated_patient.index:
                    self.df.loc[patient_idx[0], col] = updated_patient[col]
                
                # Save the updated dataframe
                self.df.to_csv(self.data_file, index=False)
                
                # Show success message
                messagebox.showinfo("Success", "Patient information updated successfully")
                
                # Close edit window
                edit_window.destroy()
                
                # Close parent window
                parent_window.destroy()
                
                # Update patient records view
                self.display_patient_records(self.df)
            else:
                messagebox.showerror("Error", "Patient not found in database")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving: {str(e)}")

    def save_patient_data(self, predicted_diseases):
        """Save a new patient to the database"""
        try:
            # Get form data
            name = self.name_var.get().strip()
            age = int(self.age_var.get())
            gender = self.gender_var.get()
        
            # Create new patient record
            new_patient = {
                'Name': name,
                'Age': age,
                'Gender': gender,
                'Disease_Prediction': ", ".join(predicted_diseases) if predicted_diseases else "None"
            }
        
        # Add vital signs
            for name_var, var in self.vital_vars.items():
                # Convert variable name to column name format (e.g. bp_systolic -> BP_Systolic)
                # Format varies by system - try a few approaches
                col_name = "_".join(part.capitalize() for part in name_var.split('_'))
                new_patient[col_name] = var.get()
        
            print(f"New patient data: {new_patient}")
            print(f"DataFrame columns: {self.df.columns.tolist()}")
        
        # Check if we have a dataframe
            if self.df is None:
                # Create a new dataframe with columns matching the new patient
                self.df = pd.DataFrame(columns=list(new_patient.keys()))
                print("Created new DataFrame")
        
        # Check if the patient already exists
            existing = self.df[self.df['Name'] == name]
        
            if len(existing) > 0:
                # Ask if user wants to update existing record
                update = messagebox.askyesno(
                    "Patient Exists", 
                    f"A patient named '{name}' already exists. Do you want to update their record?"
                )
            
                if update:
                    # Update existing record - safer approach with loc
                    idx = existing.index[0]
                    for col, val in new_patient.items():
                        if col in self.df.columns:
                            self.df.loc[idx, col] = val
                        else:
                            print(f"Warning: Column {col} not in DataFrame")
                    print(f"Updated existing patient {name}")
                else:
                    # Don't save
                    return
            else:
                # Add new patient - make sure to add only with existing columns
                patient_data = {}
                for col in self.df.columns:
                    if col in new_patient:
                        patient_data[col] = new_patient[col]
                    else:
                        # Use default value if column not in new patient data
                        if col in ['Name', 'Age', 'Gender', 'Disease_Prediction']:
                            patient_data[col] = new_patient.get(col, "")
                        else:
                            patient_data[col] = 0.0
                        
                # Add any new vital signs not yet in the dataframe
                for col, val in new_patient.items():
                    if col not in self.df.columns:
                        # Add new column with default values
                        self.df[col] = 0.0
                        patient_data[col] = val
            
            # Use DataFrame.loc to append the new row
                new_idx = len(self.df)
                for col, val in patient_data.items():
                    self.df.loc[new_idx, col] = val
                
                print(f"Added new patient {name}")
        
            # Save to file with explicit path
            file_path = os.path.abspath(self.data_file)
            print(f"Saving to file: {file_path}")
            self.df.to_csv(file_path, index=False)
        
            # Show success message
            messagebox.showinfo("Success", f"Patient data for {name} saved successfully")
        
            # Update patient dropdown in diet view
            self.update_diet_patient_dropdown()
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error saving patient data: {str(e)}\n{error_details}")
            messagebox.showerror("Error", f"Failed to save patient data: {str(e)}\nAttempted to save to: {self.data_file}")
    
    def load_data(self):
        """Load patient data from CSV file"""
        try:
            if os.path.exists(self.data_file):
                self.df = pd.read_csv(self.data_file)
                
                # Clean up column names if needed
                self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
                
                # Convert Disease_Prediction to string type
                if "Disease_Prediction" in self.df.columns:
                    self.df["Disease_Prediction"] = self.df["Disease_Prediction"].astype(str)
                    self.df["Disease_Prediction"] = self.df["Disease_Prediction"].replace('nan', 'None')
                elif "Diseases" in self.df.columns:
                    self.df["Diseases"] = self.df["Diseases"].astype(str)
                    self.df["Diseases"] = self.df["Diseases"].replace('nan', 'None')
                    # Rename for consistency
                    self.df = self.df.rename(columns={"Diseases": "Disease_Prediction"})
                
                print(f"Loaded {len(self.df)} patient records")
                
                # Update the disease filter dropdown
                self.update_disease_filter()
                
                # Update diet patient dropdown
                self.update_diet_patient_dropdown()
                
                # Update vital sign dropdown if vitals tab exists
                if hasattr(self, 'vitals_tab') and self.vitals_tab is not None:
                    vital_sign_columns = [col for col in self.df.columns if col not in ["Name", "Gender", "Age", "Disease_Prediction"]]
                    if hasattr(self, 'selected_vital'):
                        self.selected_vital.set(vital_sign_columns[0] if vital_sign_columns else "")
                        vital_options = vital_sign_columns
                        for child in self.vitals_tab.winfo_children():
                            if len(child.winfo_children()) > 0:
                                for dropdown in child.winfo_children():
                                    if isinstance(dropdown, ttk.Combobox):
                                        dropdown['values'] = vital_options
            else:
                # Create sample data if file doesn't exist
                self.create_sample_dataset()
        except Exception as e:
            messagebox.showerror("Data Error", f"Error loading data: {str(e)}")
            self.create_sample_dataset()

    
    def create_sample_dataset(self, num_samples=5000):
        """Create a sample dataset with 5000 patients"""
        try:
            # Generate random patient data
            data = []
            
            # Define disease conditions and their criteria
            diseases = ["Anemia", "Hypertension", "Diabetes", "Heart Disease", 
                       "Vitamin D Deficiency", "Kidney Disease", "High Cholesterol",
                       "Iron Deficiency", "Obesity", "Malnutrition"]
            
            first_names = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "James", 
                          "Linda", "Robert", "Patricia", "William", "Susan", "Richard", "Jessica",
                          "Thomas", "Jennifer", "Daniel", "Maria", "Matthew", "Lisa"]
            
            last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", 
                         "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White",
                         "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"]
            
            for i in range(num_samples):
                # Patient info
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
                age = random.randint(18, 85)
                gender = random.choice(["Male", "Female"])
                
                # Vital signs (with some correlation to make the data realistic)
                hemoglobin = max(8.0, min(18.0, np.random.normal(14.0 if gender == "Male" else 13.0, 2.0)))
                bp_systolic = max(80, min(200, np.random.normal(120, 20)))
                bp_diastolic = max(50, min(120, np.random.normal(80, 15)))
                heart_rate = max(45, min(120, np.random.normal(75, 15)))
                hba1c = max(4.0, min(10.0, np.random.normal(5.7, 1.2)))
                vitamin_d = max(10.0, min(50.0, np.random.normal(30.0, 10.0)))
                ldl = max(50.0, min(200.0, np.random.normal(100.0, 30.0)))
                iron = max(30.0, min(180.0, np.random.normal(100.0, 30.0)))
                creatinine = max(0.5, min(3.0, np.random.normal(1.0, 0.5)))
                mch = max(20.0, min(40.0, np.random.normal(29.0, 2.0)))
                mchc = max(30.0, min(38.0, np.random.normal(33.0, 2.0)))
                crp = max(0.0, min(50.0, np.random.exponential(2.0)))
                
                # Determine diseases based on vital signs
                patient_diseases = []
                
                if hemoglobin < 12.0 or iron < 60.0:
                    patient_diseases.append("Anemia")
                    
                if bp_systolic > 140 or bp_diastolic > 90:
                    patient_diseases.append("Hypertension")
                    
                if hba1c > 6.5:
                    patient_diseases.append("Diabetes")
                    
                if heart_rate < 50 or heart_rate > 100:
                    patient_diseases.append("Heart Disease")
                    
                if vitamin_d < 20.0:
                    patient_diseases.append("Vitamin D Deficiency")
                    
                if creatinine > 1.5:
                    patient_diseases.append("Kidney Disease")
                    
                if ldl > 130.0:
                    patient_diseases.append("High Cholesterol")
                
                # Add random diseases in some cases
                if random.random() < 0.1:  # 10% chance of random disease
                    potential_disease = random.choice(diseases)
                    if potential_disease not in patient_diseases:
                        patient_diseases.append(potential_disease)
                
                # Create row
                row = {
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Hemoglobin": hemoglobin,
                    "BP_Systolic": bp_systolic,
                    "BP_Diastolic": bp_diastolic,
                    "Heart_Rate": heart_rate,
                    "HbA1c": hba1c,
                    "Vitamin_D": vitamin_d,
                    "LDL": ldl,
                    "Iron": iron,
                    "Creatinine": creatinine,
                    "MCH": mch,
                    "MCHC": mchc,
                    "CRP": crp,
                    "Disease_Prediction": ", ".join(patient_diseases) if patient_diseases else "None"
                }
                
                data.append(row)
            
            # Create DataFrame
            self.df = pd.DataFrame(data)
            
            # Save to file
            self.df.to_csv(self.data_file, index=False)
            print(f"Created sample dataset with {len(self.df)} records")
            
            # Update dropdowns
            self.update_disease_filter()
            self.update_diet_patient_dropdown()
            
            # Show confirmation
            messagebox.showinfo("Sample Data Created", f"Created sample dataset with {len(self.df)} patient records")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample data: {str(e)}")
    
    def load_model(self):
        """Load or create the prediction model"""
        try:
            model_loaded = False
            if (os.path.exists(self.model_file) and 
                os.path.exists(self.scaler_file) and 
                os.path.exists(self.mlb_file)):
            
                try:
                    # Load existing model and components
                    self.model = pickle.load(open(self.model_file, 'rb'))
                    self.scaler = pickle.load(open(self.scaler_file, 'rb'))
                    self.mlb = pickle.load(open(self.mlb_file, 'rb'))
                
                    # Verify scaler is actually a StandardScaler
                    if hasattr(self.scaler, 'transform'):
                        model_loaded = True
                        print("Model and components loaded successfully")
                    else:
                        print("Scaler is not a valid StandardScaler object. Will retrain model...")
                except Exception as e:
                    print(f"Error loading model components: {str(e)}")
        
            if not model_loaded:
                # Train a new model
                self.train_model()
        except Exception as e:
            messagebox.showerror("Model Error", f"Error loading model: {str(e)}")
            # Train a new model as fallback
            self.train_model()

    def train_model(self):
        """Train a disease prediction model using the dataset"""
        try:
            if self.df is None or len(self.df) < 10:
                messagebox.showerror("Training Error", "Not enough data to train model")
                return
        
            # Show training indicator
            training_window = tk.Toplevel(self.root)
            training_window.title("Training Model")
            training_window.geometry("300x100")
            training_window.resizable(False, False)
        
            ttk.Label(
                training_window,
                text="Training prediction model...\nThis may take a moment.",
                font=('Arial', 12),
                justify=tk.CENTER
            ).pack(pady=20)
        
            training_window.update()
        
            # Prepare features (vital signs)
            feature_cols = [col for col in self.df.columns if col not in ["Name", "Gender", "Disease_Prediction"]]
            X = self.df[feature_cols].values
        
            # Prepare target (diseases)
            disease_lists = self.df['Disease_Prediction'].apply(
                lambda x: x.split(", ") if isinstance(x, str) and x != "None" else []
            ).tolist()
        
            # Use MultiLabelBinarizer for multi-label classification
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(disease_lists)
        
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        
            # Train a Random Forest model for multi-label classification
            base_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                n_jobs=-1,
                random_state=42
            )
        
            # If multi-label (more than one disease possible)
            self.model = MultiOutputClassifier(base_model)
            self.model.fit(X_scaled, y)
        
            # Save model components
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
        
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(self.mlb_file, 'wb') as f:
                pickle.dump(self.mlb, f)
        
            # Verify the scaler was saved correctly by loading it back
            try:
                test_scaler = pickle.load(open(self.scaler_file, 'rb'))
                if not hasattr(test_scaler, 'transform'):
                    print("Warning: Scaler was not saved correctly. Attempting alternative save method...")
                    joblib.dump(self.scaler, self.scaler_file)
            except Exception as e:
                print(f"Error verifying scaler: {str(e)}")
        
            # Close training window
            training_window.destroy()
        
            print("Model trained and saved successfully")
            messagebox.showinfo("Model Trained", "Prediction model trained successfully")
        
        except Exception as e:
            messagebox.showerror("Training Error", f"Error training model: {str(e)}")
    
    def predict_disease(self):
        """Make disease prediction based on input values"""
        # Validate required fields
        if not self.name_var.get().strip():
            messagebox.showerror("Validation Error", "Please enter the patient's name")
            return
        
        try:
            age = int(self.age_var.get())
            if age <= 0 or age > 120:
                raise ValueError("Invalid age")
        except ValueError:
            messagebox.showerror("Validation Error", "Please enter a valid age (1-120)")
            return
        
        # Collect vital signs
        vital_values = {}
        for name, var in self.vital_vars.items():
            vital_values[name] = var.get()
        
        # If no model, show error
        if self.model is None or self.scaler is None or self.mlb is None:
            messagebox.showerror("Model Error", "Prediction model not loaded. Please train the model first.")
            return
    
        # Verify scaler is correct type
        if not hasattr(self.scaler, 'transform'):
            messagebox.showerror("Model Error", "The model's scaler appears to be corrupted. Retraining model...")
            self.train_model()
            return
        
        try:
            # Get the exact feature columns the scaler expects
            # We need to extract this from the scaler itself or from our model training data
            if hasattr(self.scaler, 'feature_names_in_'):
                # For newer scikit-learn versions that store feature names
                feature_cols = self.scaler.feature_names_in_
                print(f"Feature columns from scaler: {feature_cols}")
            else:
                # Fall back to columns from the dataframe, excluding non-feature columns
                feature_cols = [col for col in self.df.columns if col not in ["Name", "Gender", "Disease_Prediction"]]
                print(f"Feature columns from dataframe: {feature_cols}")
            
                # Make sure we have the right number of features
                feature_cols = feature_cols[:16]  # Limit to the 16 features the scaler expects
                print(f"Using first 16 feature columns: {feature_cols}")
        
            # Build feature vector with the correct mapping
            features = []
            for col in feature_cols:
                if col == "Age":
                    features.append(age)
                # Handle different cases for vital sign names with consistent mapping
                elif col.lower() == "hemoglobin":
                    features.append(vital_values["hemoglobin"])
                elif col.lower() in ["bp_systolic", "bp systolic", "bpsystolic"]:
                    features.append(vital_values["bp_systolic"])
                elif col.lower() in ["bp_diastolic", "bp diastolic", "bpdiastolic"]:
                    features.append(vital_values["bp_diastolic"]) 
                elif col.lower() in ["heart_rate", "heart rate", "heartrate"]:
                    features.append(vital_values["heart_rate"])
                elif col.lower() in ["hba1c", "hb a1c"]:
                    features.append(vital_values["hba1c"])
                elif col.lower() in ["vitamin_d", "vitamin d", "vitamind"]:
                    features.append(vital_values["vitamin_d"])
                elif col.lower() == "ldl":
                    features.append(vital_values["ldl"])
                elif col.lower() == "iron":
                    features.append(vital_values["iron"])
                elif col.lower() == "creatinine":
                    features.append(vital_values["creatinine"])
                elif col.lower() in ["mch", "mean corpuscular hemoglobin"]:
                    features.append(vital_values["mch"])
                elif col.lower() in ["mchc", "mean corpuscular hemoglobin concentration"]:
                    features.append(vital_values["mchc"])
                elif col.lower() in ["crp", "c reactive protein"]:
                    features.append(vital_values["crp"])
                # Add any new columns we discovered in the dataset
                elif col.lower() == "mcv":
                    features.append(vital_values.get("mcv", 90.0))  # Default value if not provided
                elif col.lower() == "uibc":
                    features.append(vital_values.get("uibc", 240.0))  # Default value
                elif col.lower() == "sgot":
                    features.append(vital_values.get("sgot", 25.0))  # Default value
                elif col.lower() == "uric_acid":
                    features.append(vital_values.get("uric_acid", 5.0))  # Default value
                else:
                    # For any unmapped column, use a reasonable default value
                    features.append(0.0)
                
            # Convert to numpy array and reshape
            features = np.array(features).reshape(1, -1)
            print(f"Feature array shape: {features.shape}, should match expected feature count")
        
            # Scale features
            scaled_features = self.scaler.transform(features)
        
            # Make prediction
            prediction = self.model.predict(scaled_features)
        
            # Convert prediction to disease names
            predicted_diseases = []
            for i, val in enumerate(prediction[0]):
                if val == 1 and i < len(self.mlb.classes_):
                    predicted_diseases.append(self.mlb.classes_[i])
        
            # Get probabilities
            disease_probs = []
            for i, disease in enumerate(self.mlb.classes_):
                if hasattr(self.model, 'estimators_') and i < len(self.model.estimators_):
                    estimator = self.model.estimators_[i]
                    if hasattr(estimator, 'predict_proba'):
                        prob = estimator.predict_proba(scaled_features)[0][1]
                        disease_probs.append((disease, prob))
        
            # Sort diseases by probability
            disease_probs.sort(key=lambda x: x[1], reverse=True)
        
            print(f"Predicted diseases: {predicted_diseases}")
            print(f"Disease probabilities: {disease_probs[:5]}")  # Show top 5 for brevity
        
            # Display prediction results
            self.display_prediction_results(predicted_diseases, disease_probs)
        
            # Generate diet recommendations
            self.display_diet_recommendations(predicted_diseases)
        
            # Save patient data
            self.save_patient_data(predicted_diseases)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Prediction Error: {str(e)}\n{error_details}")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {str(e)}")
    
    def display_prediction_results(self, predicted_diseases, disease_probs):
        """Display disease prediction results"""
        # Clear any existing prediction frame
        if self.prediction_frame.winfo_manager():
            self.prediction_frame.pack_forget()
            
        # Pack the frame
        self.prediction_frame.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        # Clear existing content
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
            
        # Display results
        if not predicted_diseases:
            # No diseases predicted
            result_label = ttk.Label(
                self.prediction_frame,
                text="No health conditions detected. The patient appears to be healthy.",
                font=('Arial', 12),
                foreground="green",
                style='TLabel'
            )
            result_label.pack(padx=15, pady=15)
        else:
            # Create scrollable text area for results
            result_text = scrolledtext.ScrolledText(
                self.prediction_frame, 
                height=10, 
                wrap=tk.WORD, 
                font=('Arial', 11)
            )
            result_text.pack(fill=tk.BOTH, padx=15, pady=15)
            
            # Insert header
            result_text.insert(tk.END, "Predicted Health Conditions:\n\n", "header")
            result_text.tag_configure("header", font=('Arial', 12, 'bold'))
            
            # Insert each disease with probability
            for disease, prob in disease_probs:
                if prob > 0.3:  # Only show diseases with >30% probability
                    line = f"• {disease}: {prob*100:.1f}% probability\n"
                    result_text.insert(tk.END, line, "disease" if prob > 0.5 else "normal")
                    
            # Configure tags
            result_text.tag_configure("disease", foreground=self.warning_color, font=('Arial', 11, 'bold'))
            result_text.tag_configure("normal", font=('Arial', 11))
            
            # Add recommendations
            result_text.insert(tk.END, "\nRecommendations:\n", "header")
            recommendations = self.generate_recommendations(predicted_diseases)
            result_text.insert(tk.END, recommendations)
            
            # Make read-only
            result_text.config(state=tk.DISABLED)
    
    def generate_recommendations(self, diseases):
        """Generate health recommendations based on predicted diseases"""
        recommendations = ""
        
        if "Anemia" in diseases:
            recommendations += "• Consider iron supplements and include iron-rich foods in diet\n"
            recommendations += "• Schedule follow-up blood tests to monitor hemoglobin levels\n"
            
        if "Hypertension" in diseases:
            recommendations += "• Monitor blood pressure regularly\n"
            recommendations += "• Reduce sodium intake and maintain a heart-healthy diet\n"
            recommendations += "• Consider medication if lifestyle changes don't improve readings\n"
            
        if "Diabetes" in diseases:
            recommendations += "• Monitor blood glucose levels regularly\n"
            recommendations += "• Follow a controlled carbohydrate diet plan\n"
            recommendations += "• Consider consultation with an endocrinologist\n"
            
        if "Heart Disease" in diseases:
            recommendations += "• Schedule an appointment with a cardiologist\n"
            recommendations += "• Consider cardiac stress test and additional evaluations\n"
            recommendations += "• Follow a heart-healthy diet low in saturated fats\n"
            
        if "Vitamin D Deficiency" in diseases:
            recommendations += "• Take vitamin D supplements as directed\n"
            recommendations += "• Increase sun exposure (15-20 minutes daily if possible)\n"
            
        if "Kidney Disease" in diseases:
            recommendations += "• Consult a nephrologist for specialized care\n"
            recommendations += "• Monitor fluid intake and follow a kidney-friendly diet\n"
            recommendations += "• Schedule regular kidney function tests\n"
            
        if "High Cholesterol" in diseases:
            recommendations += "• Follow a low-cholesterol diet\n"
            recommendations += "• Increase physical activity\n"
            recommendations += "• Consider medication if levels remain elevated\n"
            
        if not recommendations:
            recommendations = "Maintain a balanced diet and regular exercise routine for overall health.\n"
            recommendations += "Schedule annual check-ups to monitor health status.\n"
            
        return recommendations
    
    def clear_patient_form(self):
        """Clear all fields in the new patient form"""
        # Clear name
        self.name_var.set("")
        
        # Reset age to default
        self.age_var.set("30")
        
        # Reset gender to default
        self.gender_var.set("Male")
        
        # Reset all vital signs to defaults
        for vital in self.vital_signs:
            self.vital_vars[vital["var_name"]].set(vital["default"])
            
        # Hide results frames if visible
        if self.prediction_frame.winfo_manager():
            self.prediction_frame.pack_forget()
            
        if self.diet_recommendation_frame.winfo_manager():
            self.diet_recommendation_frame.pack_forget()
            
        messagebox.showinfo("Form Cleared", "All fields have been reset to default values.")

    def update_disease_filter(self):
        """Update the disease filter dropdown with unique diseases"""
        if self.df is None:
            return
            
        # Get all unique diseases
        all_diseases = []
        for diseases_str in self.df['Disease_Prediction'].dropna():
            if diseases_str != "None":
                all_diseases.extend(diseases_str.split(", "))
        
        unique_diseases = sorted(list(set(all_diseases)))
        
        # Update dropdown values
        self.disease_filter['values'] = [""] + unique_diseases


# Main application entry point
def main():
    # Create root window
    root = tk.Tk()
    
    # Create application instance
    app = HealthPredictorApp(root)
    
    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()