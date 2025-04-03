// Load Datasets
var admin0 = ee.FeatureCollection("projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM0");
var admin1 = ee.FeatureCollection("projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM1");
var admin2 = ee.FeatureCollection("projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM2");
var ntlCollection = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21");

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
var endYear = 2021;

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

// Function to calculate the total SOL (Sum of Lights) for each district over the entire period
var calculateTotalSol = function() {
  // Calculate the sum of nighttime lights for each district over the entire period
  var summedImage = ntlFiltered.mean(); // Use mean as a proxy for total sum

  var statsCol = summedImage.reduceRegions({
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
      'country': country,
      'district': region,
    });
    return newFeature;
  });

  return results;
};

// Calculate the total SOL and export the results
var solByRegionTotal = calculateTotalSol();
print('Total Sum of Lights by District', solByRegionTotal);

// Export the results as a CSV
Export.table.toDrive({
  collection: solByRegionTotal,
  description: 'NTL_Total_by_District_2013_to_2021',
  folder: 'earthengine',
  fileNamePrefix: 'NTL_District_India_Total_2013_2021',
  fileFormat: 'CSV',
  selectors: ['country', 'district', 'total_sol']
});