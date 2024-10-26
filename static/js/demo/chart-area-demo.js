
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';
function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}

// Initialize the chart globally so it can be updated later
var ctx = document.getElementById("myAreaChart");  // Get the canvas element for the chart
var myLineChart = new Chart(ctx, {  // Create a new line chart
  type: 'line',  // Set the chart type to 'line'
  data: {
    labels: [],  // Start with empty labels (x-axis)
    datasets: [{
      label: "Sentiment Score",  // Label for the dataset
      lineTension: 0.3,  // Smoothness of the line
      backgroundColor: "rgba(78, 115, 223, 0.05)",  // Background color under the line
      borderColor: "rgba(78, 115, 223, 1)",  // Line color
      pointRadius: 3,  // Radius of the data points
      pointBackgroundColor: "rgba(78, 115, 223, 1)",  // Background color of the points
      pointBorderColor: "rgba(78, 115, 223, 1)",  // Border color of the points
      pointHoverRadius: 3,  // Radius of the points when hovered
      pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",  // Background color of points when hovered
      pointHoverBorderColor: "rgba(78, 115, 223, 1)",  // Border color of points when hovered
      pointHitRadius: 10,  // Radius of the points hit area
      pointBorderWidth: 2,  // Width of the point border
      data: [],  // Start with empty data (y-axis)
      spanGaps: false // This ensures the line breaks when there is a null value
    }],
  },
  options: {
    maintainAspectRatio: false,  // Do not maintain the aspect ratio of the chart
    layout: {
      padding: {  // Padding around the chart
        left: 10,
        right: 25,
        top: 25,
        bottom: 0
      }
    },
    scales: {
      xAxes: [{
        time: {
          unit: 'seconds'  // X-axis unit (e.g., seconds)
        },
        gridLines: {  // Configuration for the grid lines on the x-axis
          display: false,  // Do not display grid lines
          drawBorder: false  // Do not draw border lines
        },
        ticks: {
          font: {
            size: 14 // 修改 x 轴字体大小
          },
          maxTicksLimit: 10  // Limit the number of ticks on the x-axis
        }
      }],
      yAxes: [{
        ticks: {
          max: 1,  // Maximum value for the y-axis
          min: -1,  // Minimum value for the y-axis
          padding: 10,  // Padding inside the y-axis
          font: {
            size: 14 // 修改 y 轴字体大小
          },
          // Format the y-axis ticks using the number_format function
          callback: function(value, index, values) {
            return number_format(value, 1, '.', ',');
          }
        },
        gridLines: {  // Configuration for the grid lines on the y-axis
          color: "rgb(234, 236, 244)",  // Grid line color
          zeroLineColor: "rgb(234, 236, 244)",  // Color of the zero line
          drawBorder: false,  // Do not draw border lines
          borderDash: [2],  // Dashed border lines
          zeroLineBorderDash: [2]  // Dashed zero line
        }
      }],
    },
    legend: {
      display: false  // Do not display the legend
    },
    tooltips: {
      backgroundColor: "rgb(255,255,255)",  // Tooltip background color
      bodyFontColor: "#858796",  // Tooltip text color
      titleMarginBottom: 10,  // Margin below the tooltip title
      titleFontColor: '#6e707e',  // Tooltip title text color
      titleFontSize: 17,  // Tooltip title font size
      bodyFontSize: 16,   // Tooltip body font size
      borderColor: '#dddfeb',  // Tooltip border color
      borderWidth: 1,  // Tooltip border width
      xPadding: 15,  // Tooltip horizontal padding
      yPadding: 15,  // Tooltip vertical padding
      displayColors: false,  // Do not display colored boxes next to tooltips
      intersect: false,  // Show the tooltip only when hovering over a point
      mode: 'index',  // Show tooltip for all items at the index
      caretPadding: 10,  // Padding inside the tooltip caret
      callbacks: {  // Custom callbacks for the tooltips
        label: function(tooltipItem, chart) {
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          // Format the tooltip label using the number_format function
          return datasetLabel + ': ' + number_format(tooltipItem.yLabel, 1, '.', ',');
        }
      }
    }
  }
});



// Function to update the chart with new data
function updateChart_old(endTimeList, scoreList) {
    // Compare the current number of labels in the chart with the number of new end times
    const chartDataLength = myLineChart.data.labels.length;
    const newDataLength = endTimeList.length;

    // If the new data contains more points, push the missing ones to the chart
    if (newDataLength > chartDataLength) {
        const extraEndTimes = endTimeList.slice(chartDataLength);  // Get extra end times
        const extraScores = scoreList.slice(chartDataLength);      // Get extra scores

        // Loop through extra data points and push them to the chart
        for (let i = 0; i < extraEndTimes.length; i++) {
            const latestEndTime = parseFloat(extraEndTimes[i]).toFixed(2);  // Ensure 2 decimal places
            const latestScore = extraScores[i] !== null && extraScores[i] !== undefined
                ? parseFloat(extraScores[i]).toFixed(2) : null;  // Or assign a default value, like 0 or an empty string
          // const latestScore = parseFloat(extraScores[i]).toFixed(2);      // Ensure 2 decimal places


            // Add the latest time and score to the chart
            myLineChart.data.labels.push(latestEndTime);  // Push latest end time to X-axis
            myLineChart.data.datasets[0].data.push(latestScore);  // Push latest score to Y-axis
        }

        // Update the chart after adding new data points
        myLineChart.update();
    }
}

function updateChart(endTimeList, scoreList) {
    // Overwrite the entire labels and data arrays with new data
    myLineChart.data.labels = endTimeList.map(time => parseFloat(time).toFixed(2));  // Ensure 2 decimal places for time
    myLineChart.data.datasets[0].data = scoreList.map(score => score !== null && score !== undefined
        ? parseFloat(score).toFixed(2) : null);  // Ensure 2 decimal places for score or null if missing

    // Update the chart with the new data points
    myLineChart.update();
}