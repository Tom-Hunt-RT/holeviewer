# Importing required packages
import pandas as pd
import streamlit as st
import plotly.express as px

# Setting layout to be wide
st.set_page_config(layout="wide")

def loaddata():
    st.write("### Load Data")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        
        try:
            drillhole_db = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                drillhole_db = pd.read_csv(uploaded_file, encoding='latin1')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    drillhole_db = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                except UnicodeDecodeError:
                    st.error("Unable to read the file with UTF-8, Latin-1, or ISO-8859-1 encoding. Please check the file encoding.")
                    return pd.DataFrame()
        return drillhole_db
    else:
        st.warning("Please upload a file.")
        return pd.DataFrame()

# Creating a list of the column headers that I might want to filter on
def createvariables(inputdata):
    if not inputdata.empty:
        variables = inputdata.columns
        variables = list(variables)
        return variables
    else:
        st.warning("No data available to create variables.")
        return []

# Creating a list of variables based on user input (via multiselect)
def selectvariables(inputdata):
    filters = createvariables(inputdata)
    if filters:
        userselection = st.multiselect("What do you want to filter on?", options=filters)
        return userselection
    else:
        return []

# Create filters
def filterdata(filters, data):
    for i in filters:
        value_or_range_or_contains = st.radio("Value, Range, or Contains?", ("Value", "Range", "Contains"), horizontal=True, key=i)
        if value_or_range_or_contains == "Value":
            choices = data[i].unique()
            choices_with_select_all = ["Select All"] + list(choices)
            user_selection = st.multiselect(f"{i} Selection", options=choices_with_select_all)
            st.session_state[f'selected_{i}'] = user_selection
            if "Select All" in user_selection:
                data = data
            else:
                data = data[data[i].isin(user_selection)]
        elif value_or_range_or_contains == "Range":
            min_value = float(min(data[i]))
            max_value = float(max(data[i]))
            lowerbound = st.text_input(f"Set Lower Bound for {i}", value=str(min_value))
            upperbound = st.text_input(f"Set Upper Bound for {i}", value=str(max_value))
            try:
                lowerbound = float(lowerbound)
                upperbound = float(upperbound)
                if lowerbound < min_value or lowerbound > max_value:
                    st.error(f"Lower cutoff must be between {min_value} and {max_value}.")
                elif upperbound < min_value or upperbound > max_value:
                    st.error(f"Upper cutoff must be between {min_value} and {max_value}.")
                elif lowerbound >= upperbound:
                    st.error("Lower cutoff must be less than upper cutoff.")
                else:
                    st.success(f"Range successfully set from {lowerbound} to {upperbound}.")
                    data = data[(data[i] >= lowerbound) & (data[i] <= upperbound)]
            except ValueError:
                st.error(f"Please enter valid numeric values for both range bounds. i.e., between {min_value} and {max_value}.")
        elif value_or_range_or_contains == "Contains":
            contains_value = st.text_input(f"Enter value or letter for {i} to contain")
            if contains_value:
                data = data[data[i].astype(str).str.contains(contains_value, case=False, na=False)]
    return data

# Downhole plots
def createdownholeplots(data, holeid_col, from_col, to_col):
    # Allow user to select the analytes they want to plot
    selected_analytes = st.multiselect("Select variable to plot", options=data.columns, default=st.session_state.get('selected_analytes', []))
    st.session_state['selected_analytes'] = selected_analytes
    
    # Allow user to select the column by which to color the points/line segments (e.g., Lithology)
    available_color_columns = [col for col in data.columns if col not in [holeid_col, from_col, to_col]]
    selected_color_column = st.selectbox("Select Color by Column", options=available_color_columns, index=st.session_state.get('selected_color_index', 0))
    st.session_state['selected_color_index'] = available_color_columns.index(selected_color_column)
    
    # Allow user to select hover data options
    hover_data_options = st.multiselect("Select hover data", options=data.columns, default=st.session_state.get('hover_data_options', []))
    st.session_state['hover_data_options'] = hover_data_options

    # Convert columns 'from' and 'to' to numeric to avoid errors
    data[from_col] = pd.to_numeric(data[from_col], errors='coerce')
    data[to_col] = pd.to_numeric(data[to_col], errors='coerce')

    # Calculate interval midpoint
    data.loc[:, 'Interval Midpoint'] = (data[from_col] + data[to_col]) / 2
    
    # Prepare data for melting (plotting)
    id_vars = [holeid_col, from_col, to_col, 'Interval Midpoint'] + hover_data_options + [selected_color_column]
    melted_data = data.melt(id_vars=id_vars, value_vars=selected_analytes, var_name='Analyte', value_name='Result')
    
    # Create the scatter plot with lines, with color reflecting the selected variable (Lithology or similar)
    downholeplot = px.scatter(
        melted_data, 
        x='Result',  # Plot the selected analyte (e.g., Cu_pct) on the x-axis
        y='Interval Midpoint',  # Plot depth (midpoint of the interval) on the y-axis
        color=selected_color_column,  # Color each point by lithology (or another variable)
        line_group=holeid_col,  # Group by drill hole (Holeid) to create separate traces
        markers=True,  # Show markers for each depth interval
        hover_data={col: True for col in hover_data_options},  # Hover data options
        title="Downhole Plot"
    )
    
    # Update the plot layout
    downholeplot.update_yaxes(autorange='reversed')  # Reverse the y-axis for depth representation
    downholeplot.update_xaxes(title='Analyte Value (e.g., Cu_pct)')
    downholeplot.update_yaxes(title='Depth (Interval Midpoint)')
    
    # Update the plot size dynamically based on user input
    stretchy_height = st.slider("Slide to stretch the y-axis", min_value=300, max_value=5000, value=1800, step=10, key="stretchy_height")
    stretchy_width = st.slider("Slide to stretch the x-axis", min_value=300, max_value=5000, value=1800, step=10, key="stretchy_width")
    
    downholeplot.update_layout(
        height=stretchy_height, 
        width=stretchy_width
    )
    
    # Display the plot
    st.plotly_chart(downholeplot, key="downholeplot")



# Calculcate unique combos of values
def variabilityanalysis(data, holeid_col, from_col, to_col):
    groupby_columns = st.multiselect("Select columns to group by", options=data.columns, default=st.session_state.get('variability_groupby_columns', []))
    st.session_state['variability_groupby_columns'] = groupby_columns
    value_column = st.selectbox("Select value column to average", options=data.columns, index=st.session_state.get('variability_value_col_index', 0))
    st.session_state['variability_value_col_index'] = data.columns.get_loc(value_column)

    # Ensure that valid selections are made
    if not groupby_columns or not value_column:
        return pd.DataFrame(columns=['Combination', 'Count', 'Counts_Percentage', 'Mean Value', 'Median Value', 'Min Value', 'Max Value', 'Range'])

    # Convert the value column to numeric, coercing errors to NaN
    data[value_column] = pd.to_numeric(data[value_column], errors='coerce')

    # Drop rows with NaN values in the value column
    data = data.dropna(subset=[value_column])

    # Perform the analysis if valid selections are made
    data['unique_id'] = data[holeid_col].astype(str) + '_' + data[from_col].astype(str) + '_' + data[to_col].astype(str)
    combinations = data.groupby(groupby_columns)['unique_id'].nunique().reset_index()
    combinations = combinations.rename(columns={'unique_id': 'Count'})
    combinations['Combination'] = combinations[groupby_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    combinations["Counts_Percentage"] = (combinations["Count"] / combinations["Count"].sum()) * 100
    combinations["Mean Value"] = data.groupby(groupby_columns)[value_column].mean().values
    combinations["Median Value"] = data.groupby(groupby_columns)[value_column].median().values
    combinations["Min Value"] = data.groupby(groupby_columns)[value_column].min().values
    combinations["Max Value"] = data.groupby(groupby_columns)[value_column].max().values
    combinations["Range"] = combinations["Max Value"] - combinations["Min Value"]

    fig = px.bar(combinations, x='Combination', y='Mean Value', title=f'Mean {value_column} value with respect to {groupby_columns}', color='Counts_Percentage', color_continuous_scale='Viridis')
    fig2 = px.bar(combinations, x='Combination', y='Median Value', title=f'Median {value_column} value with respect to {groupby_columns}', color='Counts_Percentage', color_continuous_scale='Viridis')
    st.plotly_chart(fig, key="variabilityplot")
    st.plotly_chart(fig2, key="variabilityplot2")
    st.write(combinations)
    
    return combinations

# Create a sample selection assistant
def sampleselectionassistant(data, holeid_col, from_col, to_col):
    screening_method = st.selectbox("Select screening method", options=["Pre-screening (by interval)", "Post-screening (by composite)"])
    categorical_cols = st.multiselect("Select categorical variables for filtering (i.e., your subset for analysis)", options=data.columns)
    
    unique_values = {cat_col: data[cat_col].unique() for cat_col in categorical_cols}
    
    categorical_vals = {}
    for cat_col in categorical_cols:
        categorical_vals[cat_col] = st.multiselect(f"Select categorical values for {cat_col} filtering", options=unique_values[cat_col])
    
    parameter_col = st.selectbox("Select parameter to analyze (e.g., Cu_pct, K_pct, CuCN etc.)", options=data.columns)
    target_value = st.number_input(f"Enter target value for {parameter_col}", min_value=0.0)
    
    percentage_range = st.number_input("Enter allowable deviation as a percentage of target value", min_value=0.0, max_value=1000.0)
    
    apply_mass_filter = st.checkbox("Apply mass filter (define minimum mass requirement for composite)")
    if apply_mass_filter:
        required_mass = st.number_input("Enter required mass (unit agnostic)", min_value=0.0)
        mass_per_unit = st.number_input("Enter mass per unit of length (units = To - From)", min_value=0.0)
    else:
        mass_per_unit = None
    
    select_all_holeid = st.checkbox("Select all Drillholes", value=True)
    if select_all_holeid:
        selected_drillholes = data[holeid_col].unique()
    else:
        selected_drillholes = st.multiselect("Select Drillholes", options=data[holeid_col].unique())

    filtered_data = data[data[holeid_col].isin(selected_drillholes)]
    
    # Apply categorical filters in a more memory-efficient manner
    for cat_col in categorical_cols:
        if categorical_vals[cat_col]:
            filtered_data = filtered_data[filtered_data[cat_col].isin(categorical_vals[cat_col])]
    
    lower_bound = target_value - (target_value * (percentage_range / 100))
    upper_bound = target_value + (target_value * (percentage_range / 100))
    
    if screening_method == "Pre-screening (by interval)":
        # Pre-screening: Filter representative intervals
        representative_intervals = filtered_data[(filtered_data[parameter_col] >= lower_bound) & (filtered_data[parameter_col] <= upper_bound)]
        st.write(f"Number of intervals within {percentage_range}% of the target value: {representative_intervals.shape[0]}")
        representative_intervals = representative_intervals.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)

        if mass_per_unit is not None:
            # Apply mass filter
            representative_intervals['Interval_Length'] = representative_intervals[to_col] - representative_intervals[from_col]
            representative_intervals['Mass'] = representative_intervals['Interval_Length'] * mass_per_unit
            representative_intervals['Mass'] = pd.to_numeric(representative_intervals['Mass'], errors='coerce')

            # Efficient composite interval creation using apply() instead of iterrows()
            composite_intervals = []
            current_composite = []
            current_mass = 0

            def create_composite_interval(row):
                nonlocal current_composite, current_mass
                if current_composite and row[holeid_col] == current_composite[-1][holeid_col] and row[from_col] == current_composite[-1][to_col]:
                    current_composite.append(row)
                    current_mass += row['Mass']
                else:
                    if current_composite:
                        avg_parameter_value = pd.Series([r[parameter_col] for r in current_composite]).mean()
                        composite_intervals.append({
                            'HoleID': current_composite[0][holeid_col],
                            'From': current_composite[0][from_col],
                            'To': current_composite[-1][to_col],
                            'Total_Mass': current_mass,
                            'Average_Parameter': avg_parameter_value
                        })
                    current_composite = [row]
                    current_mass = row['Mass']
                return row

            representative_intervals.apply(create_composite_interval, axis=1)

            if current_composite:
                avg_parameter_value = pd.Series([r[parameter_col] for r in current_composite]).mean()
                composite_intervals.append({
                    'HoleID': current_composite[0][holeid_col],
                    'From': current_composite[0][from_col],
                    'To': current_composite[-1][to_col],
                    'Total_Mass': current_mass,
                    'Average_Parameter': avg_parameter_value
                })

            composite_df = pd.DataFrame(composite_intervals)
            valid_composites = composite_df[composite_df['Total_Mass'] >= required_mass]
            st.write("### Valid Composites meeting the required mass:")
            st.write(valid_composites)
        else:
            st.write("### Representative Intervals based on selection method:")
            st.write(representative_intervals)
    
    elif screening_method == "Post-screening (by composite)":
        # Post-screening: Sort and process composite intervals
        representative_intervals = filtered_data.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)

        if mass_per_unit is not None:
            representative_intervals['Interval_Length'] = representative_intervals[to_col] - representative_intervals[from_col]
            representative_intervals['Mass'] = representative_intervals['Interval_Length'] * mass_per_unit
            representative_intervals['Mass'] = pd.to_numeric(representative_intervals['Mass'], errors='coerce')

            composite_intervals = []
            current_composite = []
            current_mass = 0

            def create_composite_interval(row):
                nonlocal current_composite, current_mass
                if current_mass + row['Mass'] >= required_mass:
                    while current_mass + row['Mass'] >= required_mass:
                        remaining_mass = required_mass - current_mass
                        if remaining_mass > 0:
                            current_composite.append(row)
                            current_mass += row['Mass']
                            avg_parameter_value = pd.Series([r[parameter_col] for r in current_composite]).mean()
                            composite_intervals.append({
                                'HoleID': current_composite[0][holeid_col],
                                'From': current_composite[0][from_col],
                                'To': current_composite[-1][to_col],
                                'Total_Mass': current_mass,
                                'Average_Parameter': avg_parameter_value
                            })
                            current_composite = []
                            current_mass = 0
                        else:
                            current_composite.append(row)
                            current_mass += row['Mass']
                else:
                    current_composite.append(row)
                    current_mass += row['Mass']
                return row

            representative_intervals.apply(create_composite_interval, axis=1)

            if current_composite and current_mass >= required_mass:
                avg_parameter_value = pd.Series([r[parameter_col] for r in current_composite]).mean()
                composite_intervals.append({
                    'HoleID': current_composite[0][holeid_col],
                    'From': current_composite[0][from_col],
                    'To': current_composite[-1][to_col],
                    'Total_Mass': current_mass,
                    'Average_Parameter': avg_parameter_value
                })

            composite_df = pd.DataFrame(composite_intervals)

            valid_composites = composite_df[(composite_df['Average_Parameter'] >= lower_bound) & 
                                            (composite_df['Average_Parameter'] <= upper_bound) & 
                                            (composite_df['Total_Mass'] >= required_mass)]
            st.write("### Valid Composites meeting the required mass and parameter criteria:")
            st.write(valid_composites)
        else:
            st.write("### Representative Intervals based on selection method:")
            st.write(representative_intervals)


# Create a scatter plot based on variables of interest to user
def scatteranalysis(data):
    x_variable = st.selectbox("X-axis variable", options=data.columns, index=st.session_state.get('scatter_x_index', 0), key="scatterx")
    st.session_state['scatter_x_index'] = data.columns.get_loc(x_variable)
    y_variable = st.selectbox("Y-axis variable", options=data.columns, index=st.session_state.get('scatter_y_index', 0), key="scattery")
    st.session_state['scatter_y_index'] = data.columns.get_loc(y_variable)
    colour_selection = st.selectbox("Colour selection", options=data.columns, index=st.session_state.get('scatter_color_index', 0))
    st.session_state['scatter_color_index'] = data.columns.get_loc(colour_selection)
    trend_value = "ols" if st.checkbox("Select for ordinary least squares trendline") else None
    scatterplot = px.scatter(data, x=x_variable, y=y_variable, trendline=trend_value, color=colour_selection, title=f"Scatter plot of {x_variable} vs {y_variable}")
    st.plotly_chart(scatterplot, key="scatterplot")

# Create a box plot based on variables of interest to user
def boxplot(data):
    x_variable = st.selectbox("X-axis variable", options=data.columns, index=st.session_state.get('box_x_index', 0), key="boxx")
    st.session_state['box_x_index'] = data.columns.get_loc(x_variable)
    y_variable = st.selectbox("Y-axis variable", options=data.columns, index=st.session_state.get('box_y_index', 0), key="boxy")
    st.session_state['box_y_index'] = data.columns.get_loc(y_variable)
    colour_selection = st.selectbox("Colour selection", options=data.columns, index=st.session_state.get('box_color_index', 0), key="colourselectbox")
    st.session_state['box_color_index'] = data.columns.get_loc(colour_selection)
    userboxplot = px.box(data, x=x_variable, y=y_variable, title=f"Box plot of {x_variable} vs {y_variable}", color=colour_selection)
    st.plotly_chart(userboxplot, key="userboxplot")

# Defining the main execution function
def main():
    with st.sidebar:
        st.title("Drillhole Database Analytics")
        st.cache_data()
        drillholedata = loaddata()
        if not drillholedata.empty:
            holeid_col = st.selectbox("Select your data's 'Drillhole ID' column", options=drillholedata.columns, index=st.session_state.get('holeid_col_index', 0))
            st.session_state['holeid_col_index'] = drillholedata.columns.get_loc(holeid_col)
            from_col = st.selectbox("Select you data's 'From' column", options=drillholedata.columns, index=st.session_state.get('from_col_index', 0))
            st.session_state['from_col_index'] = drillholedata.columns.get_loc(from_col)
            to_col = st.selectbox("Select your data's 'To' column", options=drillholedata.columns, index=st.session_state.get('to_col_index', 0))
            st.session_state['to_col_index'] = drillholedata.columns.get_loc(to_col)
            st.write("### Filter Data (Prior to Analysis)")
            st.write("This is not a substitute for data cleaning. Please ensure your data is clean and formatted correctly.")

            selectedvariables = selectvariables(drillholedata)

            if selectedvariables:
                st.cache_data()
                user_filtered_data = filterdata(selectedvariables, drillholedata)
            else:
                user_filtered_data = pd.DataFrame()
                st.text("Data will appear once selected")
        else:
            selectedvariables = []
            user_filtered_data = pd.DataFrame()
    
    if not selectedvariables:
        st.warning("Please upload a file and select at least one variable to filter on. If you want everything, select 'HoleID' (or equivalent), then 'Select All'.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Downhole Plot", "Interval Variability Analysis", "Scatter Plot", "Box Plot", "Sample Selection Assistant"])
        
        with st.expander("Show Filtered Data"):
            st.header("Filtered Data Display")
            st.write(user_filtered_data)
        
        if not user_filtered_data.empty:
            with tab1:
                st.header("Downhole Line Plot")
                try:
                    createdownholeplots(user_filtered_data, holeid_col, from_col, to_col)
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab2:
                try:
                    st.header("Interval Variability Analysis")
                    variabilityanalyses = variabilityanalysis(user_filtered_data, holeid_col, from_col, to_col)
                    if variabilityanalyses:
                        st.write(f"Number of Intervals Remaining: {variabilityanalyses['Count'].sum()}")
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab3:
                try:
                    st.header("Scatter Analysis")
                    scatteranalysis(user_filtered_data.reset_index(drop=True))
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab4:
                try:
                    st.header("Box Plot")
                    boxplot(user_filtered_data)
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab5:
                try:
                    st.header("Sample Selection Assistant")
                    sampleselectionassistant(user_filtered_data, holeid_col, from_col, to_col)
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")

    with st.expander("Help"):
        st.write("""
        ## How to Use This Application

        This application allows you to analyze drillhole data through various plots and analyses. Here is a step-by-step guide on how to use it:

        1. **Upload Data**: Use the sidebar to upload your drillhole data file. The file should be in CSV format.
        2. **Filter Data**: Select the variables you want to filter on and apply the desired filters. This will be used in all subsequent analyses.
        3. **Select Analysis**: Choose the type of analysis you want to perform:
            - **Downhole Line Plot**: Visualize the data along/down the drillhole.
            - **Interval Variability Analysis**: Analyze the variability of intervals with respect to different parameters (e.g., lithology and alteration types).
            - **Scatter Analysis**: Create scatter plots to visualize relationships between variables.
            - **Box Plot**: Create box plots to visualize the distribution of variables.
        4. **Sample Selection Assistant**: Use this tool to assist in selecting samples based on various criteria (e.g., mass requirements and cut off grade).

        ## What This Application Can Do

        - Load and display drillhole data from a CSV file.
        - Filter data based on user-selected criteria.
        - Generate downhole line plots, scatter plots, and box plots.
        - Perform interval variability analysis.
        - Assist in sample selection based on user-defined parameters.

        ## What This Application Can't Do

        - Handle non-CSV file formats.
        - Automatically detect and correct data quality issues.
        - Perform advanced statistical analyses beyond the provided plots and analyses.

        ## Potential Issues and How to Avoid Them

        - **File Upload Issues**: Ensure the file is in CSV format and encoded in UTF-8, Latin-1, or ISO-8859-1.
        - **Data Quality**: Ensure the data is clean and properly formatted. Missing or non-numeric values in critical columns can cause errors.
        - **Filter Selection**: Be cautious when applying multiple filters, as overly restrictive filters may result in no data being displayed.

        If you encounter any issues, please refer to the error log for more details.
        """)

# Having script execute as per convention
if __name__ == "__main__":
    main()
