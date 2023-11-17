import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from PIL import Image
import os

# Set the page layout FIRST
st.set_page_config(
    page_title="Forest Data Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load your DataFrame
excel_file_path = 'RegionWiseData.xlsx'


def paginator(label, items, items_per_page=10, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.

    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = 1

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), 0, 10)


def demonstrate_paginator():
    fruit_list = [
        'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
        'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
        'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
        'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
        'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ]
    for i, fruit in paginator("Select a fruit page", fruit_list):
        st.write('%s. **%s**' % (i, fruit))

@st.cache_resource
def load_data(excel_file_path):
    combined_df = pd.DataFrame()
    excel_file = pd.ExcelFile(excel_file_path)
    sheet_names = excel_file.sheet_names

    for sheet_name in sheet_names:
        year_df = pd.read_excel(excel_file, sheet_name=sheet_name)
        year_df['Year'] = sheet_name
        combined_df = pd.concat([combined_df, year_df], ignore_index=True)

    return combined_df


def list_images_in_folder(folder_path):
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    images = [file for file in os.listdir(folder_path) if any(file.lower().endswith(ext) for ext in image_extensions)]
    return images


# Load the region and country data
def read_countries_from_directory(directory_path):
    return [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]


# Function to read images from the specified country folder
def read_images_from_country(country_path):
    return [os.path.join(country_path, image) for image in os.listdir(country_path) if image.endswith(('.jpg', '.png'))]


if __name__ == "__main__":
    try:
        # check if the key exists in session state
        _ = st.session_state.keep_graphics
    except AttributeError:
        # otherwise set it to false
        st.session_state.keep_graphics = False

    region_df = load_data('RegionWiseData.xlsx')
    country_df_dict = pd.read_excel('country.xlsx', sheet_name=None)

    img_folder_name = "./Country_image"  # Change this to the actual path of your countries folder
    countries_img = read_countries_from_directory(img_folder_name)

    # Create Streamlit app title with style

    # Create Streamlit widgets for region and test year selection
    data_selection = st.sidebar.radio("**Select Data:**",
                                      ("Continent", "African Country", "Deforestration Visualization"))

    if data_selection == "Continent":
        # Region selection
        region_names = region_df['Region'].unique()
        selected_region = st.sidebar.selectbox("**Select Region**", region_names)

        test_years = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080]
        selected_test_year = st.sidebar.selectbox("**Select Test Year:**", test_years)

    elif data_selection == "African Country":
        # Get the names of the sheets (countries)

        # Country selection
        country_names = list(country_df_dict.keys())

        # Country selection
        selected_country = st.sidebar.selectbox("Select African Country:", country_names)

        test_years = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080]
        test_temp = list(range(20, 51))
        test_precp = list(range(600, 1201, 50))
        selected_test_year = st.sidebar.selectbox("Select Test Year:", test_years)
        selected_test_temp = st.sidebar.selectbox("Select Test Temperature(C):", test_temp)
        # selected_test_precp = st.sidebar.selectbox("Select Test Precipitation(mm):", test_precp)
    elif data_selection == "Deforestration Visualization":
        # Get the names of the sheets (countries)
        selected_country_img = st.sidebar.selectbox("Select Country:", countries_img)
        # selected_test_precp = st.sidebar.selectbox("Select Test Precipitation(mm):", test_precp)

    if st.sidebar.button("Submit"):
        st.title('Forest Data Analysis')

        for i in range(0, len(country_images), 2):
            image1 = Image.open(country_images[i])
            col1.image(image1, caption=f"{selected_country_img}  {names[j]}", use_column_width=True)
            j+=1

            if i + 1 < len(country_images):
                image2 = Image.open(country_images[i + 1])
                col2.image(image2, caption=f"{selected_country_img}  {names[j]}", use_column_width=True)
                j+=1
            elif len(country_images) > 0:
                # If there are less than four images, display them in a single column
                for i in range(len(country_images)):
                    image = Image.open(country_images[i])
                    st.image(image, caption=f"{selected_country_img}  {names[j]}", use_column_width=True)
                    j+=1
            else:
                st.warning(f"No images found for {selected_country_img}.")

    st.sidebar.write("-----------------------------")

    # if st.sidebar.button("Desertification"):
    #     st.title("Desertification")
    #     data_selection = ""
    #     intr, intr_caus, land_intro,impact,comparision,rawanda,Cameroon,zimbabwe,Nigeria,Sudan,SouthSudan,ivoryCoast,Sengal,Congo ,mes = st.tabs(["Introduction", "Causes", "Land Degredation","Impact","comparision","Rwanda","Cameroon","zimbabwe","Nigeria","Sudan","South Sudan","Ivory Coast","Sengal","Congo","Measure"])

    #     with intr:

    #         st.subheader("Introduction")
    #         st.write("""
    #         Desertification is a process of land degradation in arid, semi-arid, and dry sub-humid areas resulting from various factors, both natural and human-induced. This process involves the deterioration of land productivity, loss of vegetation cover, and changes in soil properties, ultimately leading to the transformation of once fertile land into desert-like conditions.
    #         """)

            

    #         image_intro = Image.open('desert_intro.png')
    #         st.image(image_intro,use_column_width=True)
        

    #     with intr_caus:
    #         st.subheader("Causes of Desertification")
    #         st.write("""
    #         ## Anthropogenic Causes
    #         - Unsustainable agricultural Practices
    #         - Deforestation
    #         - Overgrazing
    #         """)
    #         st.write("""
    #         ## Natural Causes
    #         - Rainfall and Drought
    #         - Climate Change
    #         - Wind Erosion 

    #         """)
    #         image_intro4 = Image.open('cause_intro1.jpeg')
    #         st.image(image_intro4, use_column_width=True)

    #     with land_intro:
    #         st.subheader("Land Degradation in World")
    #         image_intro2 = Image.open('land1.jpeg')
    #         st.image(image_intro2, use_column_width=True)
    #         st.subheader("Land Degradation in Africa")
    #         image_intro3 = Image.open('land2.jpeg')
    #         st.image(image_intro3, use_column_width=True)
    #     with impact:
    #         st.subheader("Impact of Desertification")
    #         st.write("""
    #         ## Impact of Desertification in African Continent:
    #         - Loss of livelihood and food security
    #         - Reduction of Portable water
    #         - Loss of BIodiversity 
    #         - Health Issues
    #         - Displacement and Migration 

    #         ## Case Study of Sahel Region

    #     - The Sahel region is stretched across Africa from the Atlantic Ocean to the Red Sea.
    #     - The combination of climate change, overgrazing, deforestation, and improper agricultural practices has resulted in extensive land degradation and desertification. 
    #     - Around 8 months of the  year, the weather over this region is dry and average rainfall of 100-200 mm in north sahel and  500-600 mm in south sahel.
    #     - The population growth over the years has caused illegal farming to take place over the last few years and has resulted in major soil erosion and desertification to take place. 
    #     - With increase in population of human and livestock, the dependence on forest and grasslands increased in the area resulting in the overexploitation of resources.
    #     - It is estimated  that the sahel lost and saharan desert gain increases with the avg rate of 60km2/year.

    #         """)
    #         pip11 = Image.open('./pic43.jpg')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         st.image(pip11, use_column_width=True)
    #         pip12 = Image.open('./pic42.jpg')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         st.image(pip12, use_column_width=True)
    #         pip13 = Image.open('./pic41.jpg')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         st.image(pip13, use_column_width=True)
    #     with comparision:
    #         st.subheader("Comparision")
    #         Comparision1 = Image.open('./temp_precp_comp.jpeg')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         st.image(Comparision1, use_column_width=True)
    #     with rawanda:
    #         st.subheader("Graphs for Rawanda")
    #         image_raw1 = Image.open('./charts/Rwanda1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(image_raw1, use_column_width=True)
    #         image_raw2 = Image.open('./charts/Rwanda2.png')
    #         st.image(image_raw2, use_column_width=True)
    #     with Cameroon:
    #         st.subheader("Graphs for Cameroon")
    #         cameroon1 = Image.open('./charts/cameroon1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(cameroon1, use_column_width=True)
    #         cameroon2 = Image.open('./charts/cameroon 2.png')
    #         st.image(cameroon2, use_column_width=True)
    #     with zimbabwe:
    #         st.subheader("Graphs for zimbabwe")
    #         zimbabwe1 = Image.open('./charts/zimbabwe1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(zimbabwe1, use_column_width=True)
    #         zimbabwe2 = Image.open('./charts/zimbabwe2.png')
    #         st.image(zimbabwe2, use_column_width=True)
    #     with Nigeria:
    #         st.subheader("Graphs for Nigeria")
    #         Nigeria1 = Image.open('./charts/nigeria 1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(Nigeria1, use_column_width=True)
    #         Nigeria2 = Image.open('./charts/nigeria 2.png')
    #         st.image(Nigeria2, use_column_width=True)
    #         # image_intro4 = Image.open('cause_intro1.jpeg')
    #         # st.image(image_intro4, use_column_width=True)
    #     with Sudan:
    #         st.subheader("Graphs for Sudan")
    #         Sudan1 = Image.open('./charts/sudan 1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(Sudan1, use_column_width=True)
    #         Sudan2 = Image.open('./charts/sudan 2.png')
    #         st.image(Sudan2, use_column_width=True)
    #     with SouthSudan:
    #         st.subheader("Graphs for South Sudan")
    #         southSudan1 = Image.open('./charts/south Sudan 2.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(southSudan1, use_column_width=True)
    #         southSudan2 = Image.open('./charts/south sudan 2.png')
    #         st.image(southSudan2, use_column_width=True)
    #     with ivoryCoast:
    #         st.subheader("Graphs for Ivory Coast")
    #         IvoryCoas1 = Image.open('./charts/ivory coast 1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(IvoryCoas1, use_column_width=True)
    #         IvoryCoas2 = Image.open('./charts/ivory coast 2.png')
    #         st.image(IvoryCoas2, use_column_width=True)
    #     with Sengal:
    #         st.subheader("Graphs for Senegal")
    #         Senegal1 = Image.open('./charts/senegal 1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(Senegal1, use_column_width=True)
    #         Senegal2 = Image.open('./charts/senegal 2.png')
    #         st.image(Senegal2, use_column_width=True)
    #     with Congo:
    #         st.subheader("Graphs for Congo")
    #         Cong1 = Image.open('./charts/congo 1.png')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         # st.image(Cong1, use_column_width=True)
    #         Cong2 = Image.open('./charts/congo 2.png')
    #         st.image(Cong2, use_column_width=True)
    #     with mes:
    #         st.subheader("")
    #         st.write("""
    #         ## Measure for Restricting and Reducing Desertification
    #         - Sustainable Land Management
    #         - Afforestation and Reforestation
    #         - Water Management
    #         - Drought-Resistant Crops
    #         - Community Engagement
    #         - Livestock Management
    #         - Government Support
    #         - Education and Awareness

    #             """ )
            
            
    #         st.write("""
    #         ## Green Wall Initiative in Africa:
    #             It was launched in 2007 by the African Union, Great Green Wall initiative aims to restore the continent’s degraded landscapes and transform millions of lives in the Sahel. It was implemented  across 22 African countries and would revitalize thousands of communities across the continent.
    #             Objective:
    #             - Restore 100 million hectares of currently degraded land; sequester 250 million tons of carbon and create 10 million green jobs by 2030
    #             - Reforestation and Afforestation
    #             - Biodiversity Conservation
    #             - Climate Change Mitigation
    #             - Economic Development
    #             - Community Involvement
    #             - Sustainable Land Use
    #             """
    #                 )
    #         s11 = Image.open('./s1.jpg')
    #         st.markdown("<br><br>", unsafe_allow_html=True)
    #         st.image(s11, use_column_width=True)
            
    #         if data_selection == "Continent":
    #             # Function to filter and return the DataFrame for the selected region
    #             def filter_region(selected_region):
    #                 return region_df[region_df['Region'] == selected_region]


    #             def train_linear_regression(X, y):
    #                 model = LinearRegression()
    #                 model.fit(X, y)
    #                 return model


    #             # Function to update the output when the dropdown values change
    #             def on_dropdown_change():
    #                 selected_region_df = filter_region(selected_region)

    #                 X = selected_region_df['Year'].values.reshape(-1, 1)
    #                 y = selected_region_df['Forest area (million ha)'].values

    #                 model = train_linear_regression(X, y)

    #                 test_year = np.array([[selected_test_year]])
    #                 forest_area_pred = model.predict(test_year)

    #                 test_year = np.array([[selected_test_year]])
    #                 forest_area_pred = model.predict(test_year)

    #                 st.write(f":green[Selected Region:] {selected_region}")
    #                 st.write(f":green[Selected Test Year:] {selected_test_year}")
    #                 st.write(f":green[Predicted Forest Area in {selected_test_year}:] {forest_area_pred[0]:.2f} million ha")

    #                 # Display the selected region DataFrame with a styled table
    #                 # st.subheader("Selected Region Data:")
    #                 # st.dataframe(selected_region_df.style.highlight_max(axis=0), height=500)

    #                 # Create a bar chart for Forest Area Over the Years

    #                 fig1, ax1 = plt.subplots(figsize=(10, 6))
    #                 ax1.bar(selected_region_df['Year'], selected_region_df['Forest area (million ha)'], color='green')
    #                 ax1.set_xlabel('Year')
    #                 ax1.set_ylabel('Forest Area (million ha)')
    #                 ax1.set_title(f'Forest Area Over the Years for {selected_region}', fontsize=16)

    #                 # Create a bar chart for Predicted Value vs. Test Year

    #                 fig2, ax2 = plt.subplots(figsize=(10, 6))

    #                 # Calculate the predicted values for the next 50 years
    #                 predicted_values = []
    #                 test_years_next50 = [2030, 2040, 2050, 2060]

    #                 for year in test_years_next50:
    #                     test_year = np.array([[year]])
    #                     forest_area_pred = model.predict(test_year)
    #                     predicted_values.append(forest_area_pred[0])

    #                 ax2.bar(test_years_next50, predicted_values, color='green')
    #                 ax2.set_xlabel('Test Year')
    #                 ax2.set_ylabel('Predicted Forest Area (million ha)')
    #                 ax2.set_title(f'Predicted Value vs. Test Year for {selected_region}', fontsize=16)

    #                 # Display both bar charts side by side
    #                 col1, col2 = st.columns(2)
    #                 col1.pyplot(fig1)
    #                 col2.pyplot(fig2)


    #             # Call the on_dropdown_change function when widgets change
    #             on_dropdown_change()

    #         elif data_selection == "African Country":
    #             def filter_country(selected_country):
    #                 return country_df_dict[selected_country]


    #             # Function to update the output when the dropdown values change
    #             def on_dropdown_change():
    #                 selected_country_df = filter_country(selected_country)
    #                 st.write("Forest Area Prediction using polynomial Regression")
    #                 st.write(f":green[Selected African Country:] {selected_country}")
    #                 st.write(":green[Tested value table:]")

    #                 X = selected_country_df[['Year', 'Temperature( C)']].values
    #                 y = selected_country_df['Forest Area(km²)'].values

    #                 # Split the data into training and testing sets
    #                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42,
    #                                                                     shuffle=False)

    #                 # Train the Linear Regression model
    #                 degree = 2  # You can adjust the degree of the polynomial
    #                 model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    #                 model.fit(X_train, y_train)

    #                 # Make predictions on the test set
    #                 y_pred = model.predict(X_test)
    #                 mse = mean_absolute_error(y_test, y_pred)
    #                 r2 = r2_score(y_test, y_pred)
    #                 # st.write(X_test.type)
    #                 combined_array = np.column_stack((X_test[:, 0], y_test, y_pred))

    #                 # Create a DataFrame from the combined array
    #                 df = pd.DataFrame(combined_array,
    #                                 columns=['Year', 'Actual Forest Area(km²)', 'Predicted Forest Area(km²'])
    #                 st.write(df)
    #                 st.write(f"Mean absolut Error: {mse:.2f}")
    #                 st.write(f":green[Selected Test Year:] {selected_test_year}")
    #                 st.write(f":green[Selected Test Temperature(C):] {selected_test_temp}")
    #                 # st.write(f":green[Selected Test Precipitation(mm):] {selected_test_precp}")
    #                 test_year = np.array([[selected_test_year, selected_test_temp]])
    #                 forest_area_pred = model.predict(test_year)
    #                 st.write(f":green[Predicted Forest Area in {selected_test_year}:] {forest_area_pred[0]:.2f} million ha")

    #                 # st.write(f"R-squared Score: {r2:.2f}")
    #                 # print(X_train)
    #                 # st.write(X_train)

    #                 # Display the selected country DataFrame with a styled table
    #                 st.subheader(f"{selected_country} Data:")
    #                 st.dataframe(selected_country_df[
    #                     ['Year', 'Temperature( C)', 'Precipitation(mm)', 'Forest Area(km²)']].style.highlight_max(
    #                     axis=0), height=500)

    #                 fig1, ax1 = plt.subplots(figsize=(7, 5))
    #                 ax1.bar(selected_country_df['Year'], selected_country_df['Forest Area(km²)'], color='orange')
    #                 ax1.set_xlabel('Year')
    #                 ax1.set_ylabel('Forest Area(km²)')
    #                 ax1.set_title(f'Forest Area Over the Years for {selected_country}', fontsize=16)
    #                 st.pyplot(fig1)
    #                 fig1, ax1 = plt.subplots(figsize=(7, 5))

    #                 # Use Seaborn color palette for more attractive colors

    #                 # Create a bar chart for Forest Area Over the Years


    #             # Call the on_dropdown_change function when widgets change
    #             on_dropdown_change()
    #         elif data_selection == "Deforestration Visualization":

    #             country_path = os.path.join(img_folder_name, selected_country_img)
    #             country_images = read_images_from_country(country_path)
    #             # print(country_images)
    #             # Display images for the selected country in combinations of 2 (2 rows x 2 columns)
    #             st.write("Visualization of Deforestation Through Clustering")
    #             st.write(f"**{selected_country_img} Images:**")
    #             st.write("Visualization of Deforestation Through Clustering")
    #             st.markdown("<p style='color:red'>Red Colour denotes loss of vegetation.</p>", unsafe_allow_html=True)
    #             names = []
    #             if selected_country_img == "Cameroon":
    #                 names = ["2005", "2010", "2015", "2022"]
    #             else:
    #                 names = ["2000", "2005", "2010", "2015", "2022"]
    #             j = 0
    #             if len(country_images) >= 4:
    #                 # If there are at least four images, display them in combinations of 2
    #                 col1, col2 = st.columns(2)

    #                 for i in range(0, len(country_images), 2):
    #                     image1 = Image.open(country_images[i])
    #                     col1.image(image1, caption=f"{selected_country_img}  {names[j]}", use_column_width=True)
    #                     j += 1

    #                     if i + 1 < len(country_images):
    #                         image2 = Image.open(country_images[i + 1])
    #                         col2.image(image2, caption=f"{selected_country_img}  {names[j]}", use_column_width=True)
    #                         j += 1
    #             elif len(country_images) > 0:
    #                 # If there are less than four images, display them in a single column
    #                 for i in range(len(country_images)):
    #                     image = Image.open(country_images[i])
    #                     st.image(image, caption=f"{selected_country_img}  {names[j]}", use_column_width=True)
    #                     j += 1
    #             else:
    #                 st.warning(f"No images found for {selected_country_img}.")

    # st.sidebar.write("-----------------------------")

    if st.sidebar.button("Desertification"):
        st.title("Desertification")
        data_selection = ""
        intr, intr_caus, land_intro, impact, rawanda, Cameroon, zimbabwe, Nigeria, Sudan, SouthSudan, ivoryCoast, Sengal, Congo = st.tabs(
            ["Introduction", "Causes", "Land Degredation", "Impact", "Rwanda", "Cameroon", "zimbabwe", "Nigeria",
             "Sudan",
             "South Sudan", "Ivory Coast", "Sengal", "Congo"])

        with intr:
            st.subheader("Introduction")
            st.write("""
               Desertification is a process of land degradation in arid, semi-arid, and dry sub-humid areas resulting from various factors, both natural and human-induced. This process involves the deterioration of land productivity, loss of vegetation cover, and changes in soil properties, ultimately leading to the transformation of once fertile land into desert-like conditions.
            """)

            image_intro = Image.open('desert_intro.png')
            st.image(image_intro, use_column_width=True)

        with intr_caus:
            st.subheader("Causes of Desertification")
            st.write("""
            ## Anthropogenic Causes
            - Unsustainable agricultural Practices
            - Deforestation
            - Overgrazing
            """)
            st.write("""
            ## Natural Causes
            - Rainfall and Drought
            - Climate Change
            - Wind Erosion 

            """)
            image_intro4 = Image.open('cause_intro1.jpeg')
            st.image(image_intro4, use_column_width=True)

        with land_intro:
            st.subheader("Land Degradation in World")
            image_intro2 = Image.open('land1.jpeg')
            st.image(image_intro2, use_column_width=True)
            st.subheader("Land Degradation in Africa")
            image_intro3 = Image.open('land2.jpeg')
            st.image(image_intro3, use_column_width=True)
        with impact:
            st.subheader("Impact of Desertification")
            st.write("""
            ## Impact of Desertification in African Continent:
            - Loss of livelihood and food security
            - Reduction of Portable water
            - Loss of BIodiversity 
            - Health Issues
            - Displacement and Migration 

            """)

        with rawanda:
            st.subheader("Graphs for Rawanda")
            image_raw1 = Image.open('./charts/Rwanda1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(image_raw1, use_column_width=True)
            image_raw2 = Image.open('./charts/Rwanda2.png')
            st.image(image_raw2, use_column_width=True)
        with Cameroon:
            st.subheader("Graphs for Cameroon")
            cameroon1 = Image.open('./charts/cameroon1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(cameroon1, use_column_width=True)
            cameroon2 = Image.open('./charts/cameroon 2.png')
            st.image(cameroon2, use_column_width=True)
        with zimbabwe:
            st.subheader("Graphs for zimbabwe")
            zimbabwe1 = Image.open('./charts/zimbabwe1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(zimbabwe1, use_column_width=True)
            zimbabwe2 = Image.open('./charts/zimbabwe2.png')
            st.image(zimbabwe2, use_column_width=True)
        with Nigeria:
            st.subheader("Graphs for Nigeria")
            Nigeria1 = Image.open('./charts/nigeria 1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(Nigeria1, use_column_width=True)
            Nigeria2 = Image.open('./charts/nigeria 2.png')
            st.image(Nigeria2, use_column_width=True)
            # image_intro4 = Image.open('cause_intro1.jpeg')
            # st.image(image_intro4, use_column_width=True)
        with Sudan:
            st.subheader("Graphs for Sudan")
            Sudan1 = Image.open('./charts/sudan 1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(Sudan1, use_column_width=True)
            Sudan2 = Image.open('./charts/sudan 2.png')
            st.image(Sudan2, use_column_width=True)
        with SouthSudan:
            st.subheader("Graphs for South Sudan")
            southSudan1 = Image.open('./charts/south Sudan 2.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(southSudan1, use_column_width=True)
            southSudan2 = Image.open('./charts/south sudan 2.png')
            st.image(southSudan2, use_column_width=True)
        with ivoryCoast:
            st.subheader("Graphs for Ivory Coast")
            IvoryCoas1 = Image.open('./charts/ivory coast 1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(IvoryCoas1, use_column_width=True)
            IvoryCoas2 = Image.open('./charts/ivory coast 2.png')
            st.image(IvoryCoas2, use_column_width=True)
        with Sengal:
            st.subheader("Graphs for Senegal")
            Senegal1 = Image.open('./charts/senegal 1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(Senegal1, use_column_width=True)
            Senegal2 = Image.open('./charts/senegal 2.png')
            st.image(Senegal2, use_column_width=True)
        with Congo:
            st.subheader("Graphs for Congo")
            Cong1 = Image.open('./charts/congo 1.png')
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image(Cong1, use_column_width=True)
            Cong2 = Image.open('./charts/congo 2.png')
            st.image(Cong2, use_column_width=True)

    st.sidebar.write("-----------------------------")

    st.sidebar.text("How Politics affect forest area?")

    if st.sidebar.button("Rwanda Case Study"):
        st.title("Case Study: Rwanda")
        data_selection = ""
        case, data, lst1, lst2, ref = st.tabs(["Case Study", "Data Analysis", "Landsat7", "Landsat 8", "References"])

        with case:
            st.subheader("Introduction")
            st.write("""
                Rwanda is a small landlocked country in east-central Africa. Total land area is 26.338 km2
                , which is
                around 5% of Sweden’s. It is the fourth smallest country on the African mainland after Gambia,
                Eswatini (Swaziland) and Djibouti. The country is hilly and mountainous with an altitude ranging
                between 900 m and 4.500 m above sea level. It has a tropical climate with average annual
                temperature ranging between 16°C and 20°C, without significant variation.
            """)

            st.subheader("History")
            st.write("""
                The Rwanda War (1990-1994) was a brutal ethnic conflict between the Hutu-led government and the Tutsi rebels of the RPF. 
                The war resulted in the tragic 1994 genocide, with around 800,000 people killed. The impact on forests and protected areas was significant as many of them were used for military purposes, leading to deforestation and ecosystem degradation. The war also disrupted conservation efforts, making it challenging to protect Rwanda's rich biodiversity and natural resources. 
            """)

            image = Image.open('historyRwanda.jpg')
            st.image(image, caption="NASA photos reveal destruction of 99% of rainforest park in Rwanda",
                     use_column_width=True)
            st.write("""
                NASA’s Landsat 5 satellite captured the left-side image on July 19, 1986, while NASA’s Landsat 7 satellite captured the right-side image on December 11, 2001. 
                Densely forested areas are deep green.
            """)

            st.subheader("Journey of Political Stability")
            st.write("""
                Rwanda has made significant strides in achieving political stability since the end of the Rwanda War and the genocide in 1994.

                President Paul Kagame, who has been in power since 2000, has played a central role in stabilizing the country. His government implemented a range of policies aimed at reconciliation, nation-building, and good governance.

                Rwanda's economy has experienced significant growth, partly due to prudent economic policies and investments in various sectors, including agriculture, tourism, and technology. Economic development has contributed to greater stability.

                The government promoted a sense of Rwandan national identity, discouraging the use of divisive ethnic labels (Hutu, Tutsi, and Twa) and encouraging unity.
                Rwanda adopted a unique system of power-sharing, where no ethnic group dominates in government positions. This has helped to ensure a more inclusive political environment.

            """)

            st.subheader("How Rwanda became a restoration leader")

            st.write("""
                Thanks to Rwanda's vision and forward-thinking laws and regulations, the country became one of the early adopters of the Bonn Challenge - a global effort to bring 150 million hectares of the world’s deforested and degraded land into restoration by 2020, and 350 million hectares by 2030. 
            """)

            st.write("""In 2011, Rwanda introduced a Green Growth and Climate Resilience Strategy to guide the country to become a developed, low carbon economy by 2050. Recognising the importance of the country’s forests to this goal, the strategy features ‘Sustainable Forests and Agroforestry’ as one of 14 programmes of action.
            \nThe ‘Sustainable Forests and Agroforestry’ programme aims to ensure effective and long-term management of both individual and government-owned forests.
            Today, 30.4% of Rwanda is covered with forests. Furthermore, every year Rwandans plant millions of trees as part of an annual tree planting season with the common objective to maintain the forest coverage and increase their productivity.""")

            st.write("""
                Under the country’s privatisation policy, Rwanda has partnered with the private sector to ensure the efficient management of the state-owned forests. The Ministry of Environment launched the first Private Forest Management Units in 2019 to safeguard individual forests and boost forest harvesting as a strategy to maintain and manage woodlots effectively. 
                Today, 23,456.15 hectares (equivalent to 38.4% of state forests) are now managed by private investors through long term concession agreements.
            """)

            st.subheader("Reaping rewards: Nature-based Tourism")

            st.write("""
                Rwanda’s GDP grew by 8.4 percent in the first three quarters of 2022, after reaching 11 percent in 2021. Growth was spurred by the services sector, especially the revival of tourism, leading to the improvement of employment indicators to levels similar to those at the beginning of the COVID-19 pandemic in early 2020.
            """)

            st.write("""
                Tourism is a major source of Rwanda’s foreign exchange earnings and tends to generate a higher proportion of formal sector jobs than other sectors. 

                Within the tourism sector, nature-based tourism, which accounts for 80 percent of leisure and business visitors in Rwanda, not only helps protect biodiversity and advance Rwanda’s efforts to adapt to climate change, but also plays an important role in job creation: for every $1 million (about Rwf 1,050 million) that nature-based tourism activities inject into the economy, it is estimated that an additional 1,328 new jobs could be created.   
            """)

            image2 = Image.open('rwandaTourism.png')
            st.image(image2, caption="Things to do in Rwanda")

            st.write("""
                Visit Rwanda is Arsenal Football Club’s official Tourism Partner and its first shirt sleeve partner. The Visit Rwanda logo features on the left sleeve of all AFC teams for the duration of the exciting, three-year partnership.

                The Arsenal shirt is seen 35 million times a day globally and AFC is one of the most watched teams around the world, enabling Visit Rwanda to be seen in football-loving nations around the world and helping its drive to be an even more successful tourism and investment destination.
            """)

            st.video("https://www.youtube.com/watch?v=2CVcuL_79Ac&pp=ygUUdmlzaXQgcndhbmRhIGFyc2VuYWw%3D")

        with data:
            image4 = Image.open('rwandaInit.png')
            st.image(image4, use_column_width=True)

            image5 = Image.open('rwanda2.png')
            st.image(image5)

        with lst1:
            images = list_images_in_folder("Landsat7")
            paths = []
            for image in images:
                image_path = os.path.join("Landsat7", image)
                paths.append(image_path)
            # st.image(image_path, caption=image, width=500)
            image_iterator = paginator("Select", paths)
            indices_on_page, images_on_page = map(list, zip(*image_iterator))
            st.image(images_on_page, width=300, caption=paths[:-1])

        with lst2:
            images = list_images_in_folder("Landsat8")
            paths = []
            for image in images:
                image_path = os.path.join("Landsat8", image)
                paths.append(image_path)
            # st.image(image_path, caption=image, width=500)
            image_iterator = paginator("Select", paths)
            indices_on_page, images_on_page = map(list, zip(*image_iterator))
            st.image(images_on_page, width=300, caption=paths[:-1])

        with ref:
            st.write("https://www.visitrwanda.com/arsenal/about-the-partnership/")
            st.write("https://www.go2africa.com/destinations/rwanda/")
            st.write(
                "https://news.mongabay.com/2009/06/nasa-photos-reveal-destruction-of-99-of-rainforest-park-in-rwanda/")
            st.write("https://fra-data.fao.org/assessments/fra/2020/RWA/home/overview")
            st.write(
                "https://www.topafricanews.com/2021/06/04/wed-meet-charles-karangwa-of-iucn-to-learn-more-about-forest-landscape-restoration-program-for-rwanda-and-the-region/")
            st.write("https://www.sciencedirect.com/science/article/pii/S2351989415000141")

    # if st.sidebar.button("Zimbabwe Case Study"):
    #     st.title("Case Study: Zimbabwe")
    #     data_selection = ""
    #     case, data, ref = st.tabs(["Case Study", "Data Analysis", "References"])
    #     if case:
    #         pass
    #
    #     if data:
    #         pass
    #
    #     if ref:
    #         pass
