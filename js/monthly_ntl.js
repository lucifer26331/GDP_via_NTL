// Load Datasets (Merged Monthly)
var admin0 = ee.FeatureCollection("projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM0");
var admin1 = ee.FeatureCollection("projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM1");
var admin2 = ee.FeatureCollection("projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM2");
var ntlCollection = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21");

// Apply Filters

// Select the country India
var country = 'India';
var selectedAdmin0 = admin0.filter(ee.Filter.eq('shapeName', country));
var shapeID = ee.Feature(selectedAdmin0.first()).get('shapeID');

// Filter districts (Admin2) within India
var admin2Filtered = admin2.filter(ee.Filter.eq('ADM0_shape', shapeID));
var geometry = admin2Filtered.geometry();
Map.centerObject(geometry, 5);
print('Selected Admin2 Districts', admin2Filtered);

// Set the date range from 2013 to 2021
var startYear = 2013;
var endYear = 2014;

var startDate = ee.Date.fromYMD(startYear, 1, 1);
var endDate = ee.Date.fromYMD(endYear + 1, 1, 1);

var band = 'average';

var ntlFiltered = ntlCollection
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.bounds(geometry))
  .select(band);
print('Filtered NTL collection', ntlFiltered);

var projection = ntlFiltered.first().projection();
var resolution = projection.nominalScale();
print('NTL Image Resolution', resolution);

// Function to calculate SOL (Sum of Lights) for each month
var calculateMonthlySol = function(image) {
  var date = ee.Date(image.get('system:time_start'));
  var year = date.get('year');
  var month = date.get('month');
  
  // Calculate the sum of nighttime lights for each district
  var statsCol = image.reduceRegions({
    collection: admin2Filtered,
    reducer: ee.Reducer.sum().setOutputs(['total_sol']),
    scale: resolution,
    tileScale: 16
  });

  // Process the results
  var results = statsCol.map(function(feature) {
    var solTotal = ee.List([feature.getNumber('total_sol'), 0])
    .reduce(ee.Reducer.firstNonNull());

    var country = feature.get('shapeGroup');
    var region = feature.get('shapeName');
    var newFeature = ee.Feature(null, {
      'total_sol': solTotal,
      'year': year,
      'month': month,
      'country': country,
      'district': region,
    });
    return newFeature;
  });
  
  return results;
};

// Apply the function on all images and flatten the results
var solByRegionTimeSeries = ntlFiltered
  .map(calculateMonthlySol)
  .flatten();
print('Sum of Lights by District Monthly Time Series', solByRegionTimeSeries);

// Export the results as a CSV
Export.table.toDrive({
  collection: solByRegionTimeSeries,
  description: 'NTL_Time_Series_by_District_Monthly_2013_to_2021',
  folder: 'earthengine',
  fileNamePrefix: 'NTL_District_India_Monthly_2013_2021',
  fileFormat: 'CSV',
  selectors: ['country', 'district', 'year', 'month', 'total_sol']
});