import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shutil
import zstandard as zstd


with open('rta_model_deploy_c.joblib', 'rb') as f_in:
    cctx = zstd.ZstdCompressor(level=10)
    with open('my_compress.joblib.zst', 'wb') as f_out:
        writer = cctx.stream_writer(f_out)
        writer.write(f_in.read())
        writer.flush(zstd.FLUSH_FRAME)
        
        
with open('my_compress.joblib.zst', 'rb') as compressed_file:
    # Create a decompression context
    dctx = zstd.ZstdDecompressor()

    # Create a decompression stream
    with dctx.stream_reader(compressed_file) as reader:
        # Open a new file for writing the decompressed data
        with open('decompressed_new_file.joblib', 'wb') as decompressed_file:
            # Decompress the data and write it to the output file
            decompressed_file.write(reader.read())


# splitted_filenames = ["rta_model_deploy.joblib.partaa", "rta_model_deploy.joblib.partab", "rta_model_deploy.joblib.partac","rta_model_deploy.joblib.partad", "rta_model_deploy.joblib.partae", "rta_model_deploy.joblib.partaf","rta_model_deploy.joblib.partai", "rta_model_deploy.joblib.partaj", "rta_model_deploy.joblib.partak","rta_model_deploy.joblib.partal", "rta_model_deploy.joblib.partam"]


# with open('model.joblib', 'wb') as outfile:
#     for f in splitted_filenames:
#         with open(f, 'rb') as infile:
#             shutil.copyfileobj(infile, outfile)

model = joblib.load("decompressed_new_file.joblib")
encoder = joblib.load("ordinal_encoder.joblib")

st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config(page_title="Accident Severity Prediction",
                page_icon="🚧", layout="wide")

#creating option list for dropdown menu
options_city = ['Aiken',
 'Albany',
 'Alexandria',
 'Altoona',
 'Ambler',
 'Anaheim',
 'Anderson',
 'Antioch',
 'Apopka',
 'Auburn',
 'Austin',
 'Aventura',
 'Bakersfield',
 'Baton Rouge',
 'Bell Gardens',
 'Bend',
 'Bloomington',
 'Boiling Springs',
 'Boise',
 'Bonita Springs',
 'Bozeman',
 'Bradenton',
 'Brandon',
 'Brentwood',
 'Bronx',
 'Brooklyn',
 'Castro Valley',
 'Centreville',
 'Charleston',
 'Charlotte',
 'Chattanooga',
 'Chester',
 'Chicago',
 'Chico',
 'Chiloquin',
 'Cincinnati',
 'City Of Industry',
 'Clearwater',
 'Columbia',
 'Commerce',
 'Compton',
 'Conway',
 'Coos Bay',
 'Coral Gables',
 'Corona',
 'Crescent City',
 'Cutler Bay',
 'Dallas',
 'Dayton',
 'Denver',
 'Doral',
 'Eagle Point',
 'Easley',
 'Elgin',
 'Elkins Park',
 'Escondido',
 'Fairfax',
 'Fallbrook',
 'Flint',
 'Florence',
 'Fontana',
 'Fort Lauderdale',
 'Fort Mill',
 'Fort Myers',
 'Fort Walton Beach',
 'Fort Washington',
 'Frederick',
 'Fredericksburg',
 'Fresno',
 'Gainesville',
 'Garden Grove',
 'Gardena',
 'Garner',
 'Glen Allen',
 'Glendale',
 'Grand Rapids',
 'Grants Pass',
 'Grass Valley',
 'Greenville',
 'Greenwood',
 'Greer',
 'Gurnee',
 'Hacienda Heights',
 'Hanford',
 'Hawthorne',
 'Hayward',
 'Henrico',
 'Hialeah',
 'Hixson',
 'Hollister',
 'Homestead',
 'Houston',
 'Hudson',
 'Huntingdon Valley',
 'Huntington Park',
 'Inglewood',
 'Isleton',
 'Jacksonville',
 'Kalispell',
 'Kissimmee',
 'Klamath Falls',
 'La Pine',
 'La Puente',
 'Lake City',
 'Lake Elsinore',
 'Lake Forest',
 'Lake Zurich',
 'Lancaster',
 'Land O Lakes',
 'Lansdale',
 'Largo',
 'Leesburg',
 'Lehigh Acres',
 'Lexington',
 'Linden',
 'Little Rock',
 'Lodi',
 'Long Beach',
 'Los Angeles',
 'Los Gatos',
 'Louisville',
 'Lutz',
 'Madera',
 'Madison',
 'Madras',
 'Malibu',
 'Manassas',
 'Manor',
 'Marysville',
 'Melbourne',
 'Merced',
 'Meriden',
 'Meridian',
 'Miami',
 'Miami Gardens',
 'Miami Lakes',
 'Midlothian',
 'Milton',
 'Minneapolis',
 'Mission Viejo',
 'Modesto',
 'Myrtle Beach',
 'Napa',
 'Nashville',
 'Nevada City',
 'New Castle',
 'New Orleans',
 'New Port Richey',
 'New York',
 'Newark',
 'Norristown',
 'North Chesterfield',
 'North Fort Myers',
 'North Miami',
 'North Miami Beach',
 'Oakdale',
 'Oakland',
 'Ocala',
 'Ogden',
 'Oklahoma City',
 'Ontario',
 'Ooltewah',
 'Opa Locka',
 'Orange',
 'Orange Park',
 'Orangeburg',
 'Orlando',
 'Oroville',
 'Oxnard',
 'Palm Harbor',
 'Palmdale',
 'Palmetto Bay',
 'Panama City',
 'Pensacola',
 'Perris',
 'Petaluma',
 'Philadelphia',
 'Phoenix',
 'Piedmont',
 'Pinellas Park',
 'Pittsburgh',
 'Placerville',
 'Porterville',
 'Portland',
 'Pottstown',
 'Princeton',
 'Raleigh',
 'Redlands',
 'Redmond',
 'Redwood City',
 'Richmond',
 'Riverside',
 'Riverview',
 'Rochester',
 'Rock Hill',
 'Rowland Heights',
 'Sacramento',
 'Saint Cloud',
 'Saint Paul',
 'Saint Petersburg',
 'Salem',
 'Salinas',
 'Salt Lake City',
 'San Bernardino',
 'San Diego',
 'San Francisco',
 'San Jose',
 'San Leandro',
 'San Lorenzo',
 'Sanger',
 'Santa Ana',
 'Santa Barbara',
 'Santa Clarita',
 'Santa Cruz',
 'Santa Maria',
 'Santa Rosa',
 'Sarasota',
 'Scottsdale',
 'Seaside',
 'Seattle',
 'Selma',
 'Seneca',
 'Sherman Oaks',
 'Shreveport',
 'Silver Spring',
 'Simpsonville',
 'Sonoma',
 'Sonora',
 'South Miami',
 'Spartanburg',
 'Springfield',
 'Stockton',
 'Summerville',
 'Sumter',
 'Sunny Isles Beach',
 'Syracuse',
 'Tallahassee',
 'Tampa',
 'Taylors',
 'Tempe',
 'Tooele',
 'Tracy',
 'Tucson',
 'Tulare',
 'Tulsa',
 'Tuscaloosa',
 'Tyler',
 'Utica',
 'Venice',
 'Virginia Beach',
 'Visalia',
 'Vista',
 'Wake Forest',
 'Warrenton',
 'Washington',
 'Watsonville',
 'Waukegan',
 'Wesley Chapel',
 'West Chester',
 'West Valley City',
 'Westminster',
 'Whittier',
 'Wichita Falls',
 'Willits',
 'Winchester',
 'Winter Park',
 'Woodbridge',
 'York',
 'Yuba City']


options_state = ['OH', 'IN', 'KY', 'WV', 'MI', 'PA', 'CA', 'NV', 'MN', 'TX', 'MO', 'CO', 'OK', 'LA', 'KS', 'WI', 'IA', 'MS', 'NE', 'ND', 'WY', 'SD', 'MT', 'NM', 'AR', 'IL', 'NJ', 'GA', 'FL', 'NY', 'CT', 'RI', 'SC', 'NC', 'MD', 'MA', 'TN', 'VA', 'DE', 'DC', 'ME', 'AL', 'NH', 'VT', 'AZ', 'UT', 'ID', 'OR', 'WA']

options_wind_direction = ['CALM',        
'SouthWest',    
'SouthEast',    
'NorthEast',    
'North',         
'NorthWest',     
'South',         
'West',          
'East',          
'Variable']

options_junction = [False, True]

options_traffic_signal = [False,True]

options_weather_condition = ['Clear',                    
'Cloudy',                     
'Scattered Clouds',            
'Rain',                       
'Snow',                       
'Windy',                      
'Thunderstorm',               
'Other',                        
'Light Rain with Thunder']

options_sun = ['Day','Night']


options_twilight = ['Day','Night']



# features list
features = ['City', 'State', 'Junction', 'Weather_Condition', 'Traffic_Signal',
       'Civil_Twilight', 'Sunrise_Sunset', 'Wind_Direction', 'Distance(mi)',
       'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

# take input 
st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction 🚧</h1>", unsafe_allow_html=True)
def main():
       with st.form("accident_severity_form"):
              st.subheader("Please enter the following inputs:")
              
              Distance_miles = st.slider("Length of Traffic Jam in miles:",0,100, value=0, format="%d")
              Temperature = st.slider("Temperature in Fahrenheit:",-27,190, value= -25, format="%d")
              Wind_chill = st.slider("Wind Chill in F:", -48, 190, value= -50, format="%d")
              Humditity = st.slider("Humidity percentage:",1, 100, value=0, format="%d")
              Pressure = st.slider("Pressure:",16, 60, value=0, format="%d")
              Visibility = st.slider("Visibility in miles",0, 100, value=0, format="%d")
              Wind_speed = st.slider("Wind Speed in mph",0, 1087, value=0, format="%d")
              Precipitation = st.slider("Precipitation in inches",0, 10, value=0, format="%d")
            
              state = st.selectbox("State:", options=options_state)
                
              search_letter = st.text_input("City", "").upper()

            # Filter columns
              filtered_columns = [col for col in options_city if col.startswith(search_letter)]

            # Display dropdown
              if filtered_columns:
                selected_column = st.selectbox("Select a column", filtered_columns)
                
              
              weather_cond = st.selectbox("Weather Condition", options=options_weather_condition)
              wind_direction = st.selectbox("Wind direction:", options=options_wind_direction)
              junction = st.selectbox("Wind direction:", options=options_junction)
              traffic_signal = st.selectbox("Wind direction:", options=options_traffic_signal)
              sunrise_set = st.selectbox("Wind direction:", options=options_sun)
              twilight = st.selectbox("Wind direction:", options=options_twilight)
            
    
              
              submit = st.form_submit_button("Predict")

# encode using ordinal encoder and predict
       if submit:
              input_array = np.array([state,
                                   selected_column,weather_cond,wind_direction,junction,
                                   traffic_signal,sunrise_set,twilight], ndmin=2)
              
              encoded_arr = list(encoder.transform(input_array).ravel())
              
              num_arr = [Distance_miles,Temperature,Wind_chill,Humditity,Pressure,Visibility,Wind_speed,Precipitation]
              pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)              
          
              prediction = model.predict(pred_arr)
              
              if prediction == 2:
                     st.write(f"The severity prediction is Low impact")
              elif prediction == 3:
                     st.write(f"The severity prediction is Medium Impact")
              else:
                     st.write(f"The severity prediciton is High Impact")
                  
#               st.subheader("Explainable AI (XAI) to understand predictions")  
#               shap.initjs()
#               shap_values = shap.TreeExplainer(model).shap_values(pred_arr)
#               st.write(f"For prediction {prediction}") 
#               shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0],
#                               pred_arr, feature_names=features, matplotlib=True,show=False).savefig("pred_force_plot.jpg", bbox_inches='tight')
#               img = Image.open("pred_force_plot.jpg")
#               st.image(img, caption='Model explanation using shap')
              
              st.write("Developed By: Team 12")
              
              

# post the image of the accident

a,b,c = st.columns([0.2,0.6,0.2])
with b:
  st.image("Car_crash.jpeg", use_column_width=True)




#st.markdown("Please find GitHub repository link of project: [Click Here](https://github.com/avikumart/Road-Traffic-Severity-Classification-Project)")                  
                  
if __name__ == '__main__':
   main()
    
   
                
    
                     
              

       
       


