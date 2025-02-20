# Importing required packages
import pandas as pd
import streamlit as st
import plotly.express as px

# Setting layout to be wide
st.set_page_config(layout="wide")

# Loading data to be used by application and assigning to variables
def loaddata():
    st.write("### Load Data")
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                # Read the CSV using pandas while specifying the encoding
                uploaded_file.seek(0)  # Reset the file pointer
                drillhole_db = pd.read_csv(uploaded_file, encoding=encoding)
                return drillhole_db  # Return if the file is read successfully
            except UnicodeDecodeError:
                continue  # Try next encoding if one fails
        
        # If all encodings fail, show an error
        st.error("Unable to read the file with the tested encodings. Please check the file encoding.")
        return pd.DataFrame()
    
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
        value_or_range = st.radio("Value or Range?", ("Value", "Range"), horizontal=True, key=i)
        if value_or_range == "Value":
            choices = data[i].unique()
            choices_with_select_all = ["Select All"] + list(choices)
            # Check if "Select All" is selected, and if so, select all holes
            user_selection = st.multiselect(f"{i} Selection", options=choices_with_select_all)
            if "Select All" in user_selection:
                data = data
            else:
                data = data[data[i].isin(user_selection)]
        elif value_or_range == "Range":
            min_value = float(min(data[i]))
            max_value = float(max(data[i]))
            lowerbound = st.text_input(f"Set Lower Bound for {i}", value=str(min_value))
            upperbound = st.text_input(f"Set Upper Bound for {i}", value=str(max_value))
            try:
                # Convert inputs to floats, not ints (since copper grades might not be integers)
                lowerbound = float(lowerbound)
                upperbound = float(upperbound)
                # Validate the entered values
                if lowerbound < min_value or lowerbound > max_value:
                    st.error(f"Lower cutoff must be between {min_value} and {max_value}.")
                elif upperbound < min_value or upperbound > max_value:
                    st.error(f"Upper cutoff must be between {min_value} and {max_value}.")
                elif lowerbound >= upperbound:
                    st.error("Lower cutoff must be less than upper cutoff.")
                else:
                    st.success(f"Range successfully set from {lowerbound} to {upperbound}.")
                    # Filter the data based on the cutoff grade
                    data = data[(data[i] >= lowerbound) & (data[i] <= upperbound)]
            except ValueError:
                st.error(f"Please enter valid numeric values for both range bounds. i.e., between {min_value} and {max_value}.")
    return data

# Downhole plots
def createdownholeplots(data):
    with st.expander("Downhole Plot Options", expanded=False):
        # Ensure user selects Holeid, From, and To columns
        holeid_col = st.selectbox("Select 'Drillhole ID' column", options=data.columns)
        from_col = st.selectbox("Select 'From' column", options=data.columns)
        to_col = st.selectbox("Select 'To' column", options=data.columns)        
        selected_analytes = st.multiselect("Select variable to plot", options=data.columns)
        selected_color = st.selectbox("Select Colour", options=data.columns)
        hover_data_options = st.multiselect("Select hover data", options=data.columns)
    
    # Convert 'From' and 'To' columns to numeric
    data[from_col] = pd.to_numeric(data[from_col], errors='coerce')
    data[to_col] = pd.to_numeric(data[to_col], errors='coerce')
    
    data['Interval Midpoint'] = (data[from_col] + data[to_col]) / 2
    id_vars = [holeid_col, from_col, to_col, 'Interval Midpoint']
    melted_data = data.melt(id_vars=id_vars,
                            value_vars=selected_analytes,
                            var_name='Analyte',
                            value_name='Result')

    downholeplot = px.line(melted_data, x='Result', y='Interval Midpoint', color=selected_color, line_group=holeid_col, markers=True, height=800,
                           facet_col='Analyte', facet_col_wrap=4,
                           hover_data={col: True for col in hover_data_options})

    downholeplot.update_yaxes(autorange='reversed')
    downholeplot.update_xaxes(matches=None)
    downholeplot.update_layout(
        xaxis_title='Results',
        yaxis_title='Interval Midpoint',
        title='Results by Drill Hole and Interval Midpoint',
        height=1500,
    )

    st.plotly_chart(downholeplot, key="downholeplot")

# Calculcate unique combos of values
def variabilityanalysis(data):
    with st.expander("Variability Analysis Options", expanded=False):
        holeid_col = st.selectbox("Select 'Drillhole ID' column for variability analysis", options=data.columns)
        groupby_columns = st.multiselect("Select columns to group by", options=data.columns)
    
    if groupby_columns:
        data['unique_id'] = data[holeid_col].astype(str) + '_' + data['From'].astype(str) + '_' + data['To'].astype(str)
        combinations = data.groupby(groupby_columns)['unique_id'].nunique().reset_index()
        combinations = combinations.rename(columns={'unique_id': 'Count'})
        combinations['Combination'] = combinations[groupby_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        combinations["Percentage"] = (combinations["Count"] / combinations["Count"].sum()) * 100
        return combinations
    else:
        return pd.DataFrame(columns=['Combination', 'Interval Count', 'Percentage of Intervals'])

# Create a scatter plot based on variables of interest to user
def scatteranalysis(data):
    x_variable = st.selectbox("X-axis variable", options=data.columns, index=0, key="scatterx")
    y_variable = st.selectbox("Y-axis variable", options=data.columns, index=0, key="scattery")
    colour_selection = st.selectbox("Colour selection", options=data.columns, index=0)
    if st.checkbox("Select for ordinary least squares trendline"):
        trend_value = "ols"
    else:
        trend_value = None
    scatterplot = px.scatter(data, x=x_variable, y=y_variable, trendline=trend_value, color=colour_selection, title=f"Scatter plot of {x_variable} vs {y_variable}")
    
    st.plotly_chart(scatterplot, key="scatterplot")

# Create a box plot based on variables of interest to user
def boxplot(data):
    x_variable = st.selectbox("X-axis variable", options=data.columns, index=0, key="boxx")
    y_variable = st.selectbox("Y-axis variable", options=data.columns, index=0, key="boxy")
    colour_selection = st.selectbox("Colour selection", options=data.columns, index=0, key="colourselectbox")
    userboxplot = px.box(data, x=x_variable, y=y_variable, title=f"Box plot of {x_variable} vs {y_variable}", color=colour_selection)
    st.plotly_chart(userboxplot, key="userboxplot")

# Defining the main execution function
def main():
    try:
        with st.sidebar:
            st.title("Drillhole Database Analytics")
            st.cache_data()
            drillholedata = loaddata()
            if not drillholedata.empty:
                selectedvariables = selectvariables(drillholedata)
                if len(selectedvariables) != 0:
                    user_filtered_data = filterdata(selectedvariables, drillholedata)
                    variabilityanalyses = variabilityanalysis(user_filtered_data)
                    st.write(variabilityanalyses)
                    st.write(f"Number of Intervals Remaining: {variabilityanalyses['Count'].sum()}")
                else:
                    user_filtered_data = drillholedata
                    st.text("Data will appear once selected")
            else:
                user_filtered_data = pd.DataFrame() 
        col1, col2 = st.columns([1, 1])
        if not user_filtered_data.empty:
            with col1:
                downholecontainer = st.container(border=True)
                with downholecontainer:
                    st.header("Downhole Line Plot")
                    createdownholeplots(user_filtered_data)
            with col2:
                scattercontainer = st.container(border=True)
                boxplotcontainer = st.container(border=True)
                with scattercontainer:
                    st.header("Scatter Analysis")
                    scatteranalysis(user_filtered_data.reset_index(drop=True))
                with boxplotcontainer:
                    st.header("Box Plot")
                    boxplot(user_filtered_data)
            tablecontainer = st.container(border=True)
            with tablecontainer:
                st.header("Filtered Data Display")
                st.write(user_filtered_data)
    except Exception as e:
        with st.expander("**Error Log**", expanded=False):
            st.error(f"An error occurred: {e}")

# Having script execute as per convention
if __name__ == "__main__":
    main()
