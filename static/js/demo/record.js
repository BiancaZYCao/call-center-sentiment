var recordButton = document.getElementById('recordButton');
var transcriptionResult = document.getElementById('transcriptionResult');
var sentimentResult = document.getElementById('sentimentResult');
// add by me 20240903
var responseResult = document.getElementById('responseResult');
//end of code 20240903
var topicModelResult = document.getElementById('topicModelResult');

var singlishResult = document.getElementById('SinglishResult');



//added at 20240909
var startTime = null;        // 记录录音的开始时间
var timeStamps = [];         // 存储时间戳
var scores = [];             // 存储 final_score
var isRecording = false;     // 标记是否正在录音
var startFetchChart = false;

// 兼容性处理，不同浏览器可以实现getUserMedia API
// navigator.getUserMedia是用于捕获用户的音频/视频流（如麦克风录音）的 API
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
// 创建 WebSocket 连接并赋值给这个变量
var ws = null;
//record 是录音相关的对象，稍后将通过录音 API 来实例化它
var record = null;
//timeInte 用于存储定时器的 ID。这个定时器会控制录音的间隔发送操作,变量用于在录音过程中定时将音频数据片段发送到服务器，确保录音以流的形式传输
var timeInte = null;
//isRecording 变量用于追踪录音状态
var isRecording = false;
// mark previous STT speaker label
var speaker_last = 'unknown';

// Gauge configuration - 20240909
var opts = {
    angle: 0.15, // The span of the gauge arc
    lineWidth: 0.44, // The line thickness
    radiusScale: 1, // Relative radius
    pointer: {
        length: 0.6, // Relative to gauge radius
        strokeWidth: 0.035, // The thickness
        color: '#000000' // Fill color
    },
    limitMax: false, // If false, max value increases automatically if value > maxValue
    limitMin: false, // If true, the min value of the gauge will be fixed
    colorStart: '#6FADCF', // Colors start
    colorStop: '#8FC0DA', // Colors stop
    strokeColor: '#E0E0E0', // Background stroke color
    generateGradient: true, // Generate gradient
    highDpiSupport: true, // High resolution support
    staticZones: [
        { strokeStyle: "#F03E3E", min: -1, max: -0.3 }, // Red from -1 to -0.5
        { strokeStyle: "#FFDD00", min: -0.3, max: 0.3 }, // Yellow from -0.5 to 0.5
        { strokeStyle: "#30B32D", min: 0.3, max: 1 } // Green from 0.5 to 1
    ],
    staticLabels: {
        font: "12px sans-serif", // Reduced font size
        labels: [-1, -0.3, 0, 0.3, 1], // Print labels at these values
        color: "#000000", // Label text color
        fractionDigits: 1 // Numerical precision
    },
};

var target = document.getElementById('gauge'); // Your canvas element
var gauge = new Gauge(target).setOptions(opts); // Create gauge!
gauge.maxValue = 1; // Set max gauge value
gauge.setMinValue(-1);  // Set min value
gauge.animationSpeed = 32; // Set animation speed

// Set default state to neutral (0.0)
gauge.set(0.0);
document.getElementById('score').innerText = 'Score: 0.0';
document.getElementById('prediction').innerText = 'Neutral';

// Update the gauge dynamically based on API response
function updateGauge(finalScore, finalSentiment) {
    gauge.set(finalScore); // Update the gauge pointer
    document.getElementById('score').innerText = `Score: ${finalScore.toFixed(2)}`;
    document.getElementById('prediction').innerText = finalSentiment.charAt(0).toUpperCase() + finalSentiment.slice(1); // Update sentiment
}



// recordButton绑定点击事件处理函数
recordButton.onclick = function () {
    if (!isRecording) {
        startListening();
        startRecording();
    } else {
        stopRecording();
    }
};


function startRecording() {
    console.log('Start Recording');
    var speakerVerificationCheckbox = document.getElementById('speakerVerification');
    var sv = speakerVerificationCheckbox.checked ? 1 : 0;
    var lang = document.getElementById("lang").value
    // Construct the query parameters
    var queryParams = [];
    if (lang) {
        queryParams.push(`lang=${lang}`);
    }
    if (sv) {
        queryParams.push('sv=1'); //default Ture
    }
    var queryString = queryParams.length > 0 ? `?${queryParams.join('&')}` : '';

    //1. build WebSocket connection
    ws = new WebSocket(`ws://127.0.0.1:8000/ws/transcribe${queryString}`);
    ws.binaryType = 'arraybuffer';
    //2. event on channel open
    ws.onopen = function (event) {
        console.log('WebSocket connection established');
        //2.a start recording
        record.start();
        timeInte = setInterval(function () {
            if (ws.readyState === 1) {
                var audioBlob = record.getBlob();
                // console.log('Blob size: ', audioBlob.size);

                // Read the Blob content for debugging
                var reader = new FileReader();
                reader.onloadend = function () {
                    // console.log('Blob content: ', new Uint8Array(reader.result));
                    ws.send(audioBlob);
                    console.log('Sending audio data');
                    record.clear();
                };
                reader.readAsArrayBuffer(audioBlob);
            }
        }, 500);
    };
    //3. 处理服务器发送的消息
    // ws.onmessage: update UI text chart scores when msg received.
    ws.onmessage = function (evt) {
        console.log('Received message: ' + evt.data);
        try {
            var resJson = JSON.parse(evt.data);
            var textData = resJson.data;
            var speaker = resJson.speaker_label || 'unknown speaker'; // Handle missing speaker_label
            var type = resJson.type || 'unknown type';
            var timestamp = resJson.timestamp || 'no timestamp';


            // Audio Sentiment Result - 20240915
            if (type === 'audio_sentiment') {
                var audioData = JSON.parse(textData);  // Parse the JSON string in 'data'
                var finalScore = Number(audioData.final_score);  // Cast final_score to a number
                updateGauge(finalScore, audioData.final_sentiment_3);  // Update gauge
                responseResult.textContent = `Final Sentiment (3 Classes): ${audioData.final_sentiment_3}\n`;
                startFetchChart = true;
            }

            // display text content
            if (type === 'STT') {
                startFetchChart = true;
                // transcriptionResult.innerHTML += "<br><strong>" + speaker + ":</strong> " + textData;
                // if (speaker === speaker_last) {
                //     transcriptionResult.innerHTML += ' ' + textData || ' ';
                // } else {
                //     speaker_last = speaker;
                //     transcriptionResult.innerHTML += "<br><strong>" + speaker + ":</strong> " + textData;
                // }
            }
        } catch (err) {
            console.error('Failed to parse websocket message:', err);
            transcriptionResult.textContent += "\n" + evt.data;
        }

    };

    ws.onclose = function () {
        console.log('WebSocket connection closed');
    };

    ws.onerror = function (error) {
        console.error('WebSocket error: ' + error);
    };

    recordButton.textContent = "Stop Recording";
    recordButton.classList.add("recording");
    isRecording = true;
}

function startListening() {
    console.log('Start Listening to get analysis result');

    //1. Build WebSocket Connection
    wsAnalysis = new WebSocket(`ws://127.0.0.1:8000/ws/analysis`);

    // 2. Handle WebSocket connection opening
    wsAnalysis.onopen = function (event) {
        console.log('WebSocket Analysis connection established');

        // 3. Set interval to call update-chart API every second
        setInterval(function () {
            if (startFetchChart) {
                // Fetch chart updates every 1 second
                // Fetch data from the server and update chart
                fetch('http://127.0.0.1:8000/update-chart/', {
                    method: 'GET',  // Use POST method to send the request
                    headers: {
                        'Content-Type': 'application/json'  // Set the request content type to JSON
                    }
                })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok ' + response.statusText);  // Throw an error if response is not ok
                        }
                        return response.json();  // Parse the JSON response
                    })
                    .then(data => {
                        //update chart
                        var endTimeList = data.end_time;  // Extract end_time from the response data
                        var scoreList = data.final_score;  // Extract final_score from the response data
                        updateChart(endTimeList, scoreList);  // Call the function to update chart with new data
                    })
                    .catch(error => {
                        console.error('Error chart update:', error);  // Log any errors that occur during the fetch
                    });
            }
        }, 2500); // 1000ms = 1 second
    };

    // 4. Handle incoming messages from the WebSocket
    wsAnalysis.onmessage = function (event) {
        var resJson = JSON.parse(event.data);  // Parse the incoming JSON data
        var textData = resJson.data;         // Parse data content
        var processedAt = resJson.timestamp || 'no timestamp';

        // Check the type of response, process sentiment analysis result
        if (resJson.type === "text_sentiment") {
            sentimentResult.textContent = `Sentiment: ${textData}  at ${processedAt.slice(11, 19)}`;
        }
        if (resJson.type === "topics") {
            // Parse the textData to a list
            var topics = textData ? JSON.parse(textData) : null;
            console.debug('topics receieved:', topics)
            if (topics && topics.length > 0) {                
                remainingTopics = [...topics];

                // Clear the previous tags container
                const tagsContainer = document.getElementById("tags-container");
                tagsContainer.innerHTML = '';


                // Iterate over the topics array and dynamically generate and insert tags
                topics.forEach(topic => {
                    const tagElement = document.createElement('div');
                    tagElement.className = 'tag tag-primary';  // Set class name for styling (e.g., tag-success, tag-info, etc.)
                    tagElement.textContent = topic;  // Set the topic as the tag content
                                        

                    // Create a close button for each tag
                    const closeButton = document.createElement('span');
                    closeButton.className = 'close';
                    closeButton.textContent = '×';
                    closeButton.onclick = function () {
                        // Get the questions container
                        const questionsContainer = document.getElementById("questions-container");

                        // Check if the questions container exists before proceeding
                        if (!questionsContainer) {
                            console.error("Questions container does not exist.");
                            return; // Stop execution if the container does not exist
                        }

                        // Remove the topic from remainingTopics
                        remainingTopics = remainingTopics.filter(t => t !== topic);
                        console.log('Remaining topics:', remainingTopics);

                        // Remove the tag element from the container
                        tagsContainer.removeChild(tagElement);

                        // Find the corresponding topic section
                        const topicHeader = Array.from(questionsContainer.querySelectorAll('h6')).find(header => header.textContent === topic);
                        if (topicHeader) {
                            const questionDiv = topicHeader.nextElementSibling;

                            // Remove the topic header and question div from the container
                            if (questionsContainer.contains(topicHeader)) {
                                questionsContainer.removeChild(topicHeader);
                            }
                            if (questionDiv && questionsContainer.contains(questionDiv)) {
                                questionsContainer.removeChild(questionDiv);
                            }

                            // Send the updated remainingTopics to the backend
                            // sendRemainingTopicsToBackend(remainingTopics);
                        } else {
                            console.error(`Topic '${topic}' not found.`);
                        }                        

                    };
                    tagElement.appendChild(closeButton);
                    tagsContainer.appendChild(tagElement);
                });
                // If no topics have been removed, send the full list of topics to the backend
                // if (remainingTopics.length === topics.length) {
                //     sendRemainingTopicsToBackend(remainingTopics);
                // }
            }
        }

        // Handle topicsAndQuestions response
        if (resJson.type === "topicsAndQuestions") {
            var topicsAndQuestions = textData ? JSON.parse(textData) : null;
            console.debug('topicsAndQuestions received:', topicsAndQuestions);

            // // Only update if topicsAndQuestions is not empty or null
            if ((topicsAndQuestions && Object.keys(topicsAndQuestions).length > 0)) {
                
                // Clear previous content in the questions container
                const questionsContainer = document.getElementById("questions-container");
                questionsContainer.innerHTML = '';

                // Iterate over each topic and its questions
                Object.keys(topicsAndQuestions).forEach(topic => {
                    // Create a topic header
                    const topicHeader = document.createElement('h6');
                    topicHeader.textContent = topic;
                    questionsContainer.appendChild(topicHeader);

                    // Create a div to contain questions with checkboxes
                    const questionDiv = document.createElement('div');

                    topicsAndQuestions[topic].forEach(question => {                        
                        // Create a label for the checkbox and question text
                        const label = document.createElement('label');
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.name = `question-${topic}`;
                        checkbox.value = question;

                        // Add an event listener for each checkbox
                        checkbox.onchange = function () {
                            if (this.checked) {
                                // Send selected question to the backend and display selected question
                                sendQuestionToBackend(question);
                                showSelectedQuestionAnswer(question);
                            }
                        };

                        // Append the checkbox and the question text to the label
                        label.appendChild(checkbox);
                        label.appendChild(document.createTextNode(` ${question}`));

                        // Add a line break for each question
                        questionDiv.appendChild(label);
                        questionDiv.appendChild(document.createElement('br'));
                    });

                    questionsContainer.appendChild(questionDiv);
                });
            }else {
                console.debug('No topicsAndQuestions data received. Keeping previous content.');
                // Do nothing, keep the previous content if topicsAndQuestions is empty or null
            }
        }
        if (resJson.type === 'question_answer') {
            const answer = resJson.data;
            const loadingId = resJson.loadingId;

            console.log(`Answer received for loadingId ${loadingId}: ${answer}`);

            // Call displayAnswer function to update the UI
            displayAnswer(answer, loadingId);
        }
    };

    // 5. Handle WebSocket errors
    wsAnalysis.onerror = function (event) {
        console.error('WebSocket error:', event);
    };

    // 6. Handle WebSocket closure
    wsAnalysis.onclose = function (event) {
        console.log('WebSocket Analysis connection closed');
    };
}



// Function to show selected question an
function showSelectedQuestionAnswer(question) {
    const selectedList = document.getElementById("selected-list");

    // Create a new list item for the selected question
    const listItem = document.createElement('li');
    listItem.textContent = `Selected question: ${question}`;

    // Create a new list item for the loading indicator
    const loadingItem = document.createElement('li');
    loadingItem.textContent = 'Loading answer...';

    // Assign a unique ID to the loading item so it can be removed later
    const loadingId = `loading-${Math.random().toString(36).substr(2, 9)}`;
    loadingItem.id = loadingId;

    // Append the selected question and loading indicator to the list
    selectedList.appendChild(listItem);
    selectedList.appendChild(loadingItem);

    console.log(`Sending question: ${question} with loadingId: ${loadingId}`);

    // Send selected question to the backend
    sendQuestionToBackend(question, loadingId); // Pass the loadingId to track it
}

// Function to send the selected question to the backend
function sendQuestionToBackend(question, loadingId) {
    const message = {
        type: 'selected_question',
        data: question,
        loadingId: loadingId  // Pass the loadingId to track which one to remove later
    };
    wsAnalysis.send(JSON.stringify(message));  // Send the message to the backend
}



// Function to display the answer and replace the loading indicator
function displayAnswer(answer, loadingId) {
    // Find the loading element using the loadingId
    const loadingElement = document.getElementById(loadingId);

    if (loadingElement) {
        // Replace the loading indicator with the actual answer
        loadingElement.textContent = `Answer: ${answer}`;
    } else {
        console.error("Loading element not found for id:", loadingId);
    }
}



// Function to sent remaining topics to backend
function sendRemainingTopicsToBackend(remainingTopics) {
    const dataToSend = {
        type: 'remaining_topics',
        data: remainingTopics
    };

    // Print the data to the console to check the format
    console.log("Data being sent to backend:", JSON.stringify(dataToSend));

    // Send the data to the WebSocket server
    wsAnalysis.send(JSON.stringify(dataToSend));
}



function stopRecording() {
    console.log('Stop Recording');
    if (ws) {
        ws.close();
        record.stop();
        clearInterval(timeInte);
    }
    recordButton.textContent = "Start Recording";
    recordButton.classList.remove("recording");
    isRecording = false;
    startFetchChart = false;

    // Close the second WebSocket (for sentiment analysis) after 5 seconds
    setTimeout(function() {
        if (wsAnalysis) {
            console.log('Closing second WebSocket after 5 seconds');
            wsAnalysis.close();  // Close the second WebSocket connection
        }
    }, 5000);  // 5000ms = 5 seconds
}

function init(rec) {
    record = rec;
}

if (!navigator.getUserMedia) {
    alert('Your browser does not support audio input');
} else {
    navigator.getUserMedia(
        {audio: true},
        function (mediaStream) {
            init(new Recorder(mediaStream));
        },
        function (error) {
            console.log(error);
        }
    );
}

var Recorder = function(stream) {
        var sampleBits = 16; // Sample bits
        var inputSampleRate = 48000; // Input sample rate
        var outputSampleRate = 16000; // Output sample rate
        var channelCount = 1; // Single channel
        var context = new AudioContext();   // 创建Web Audio的音频上下文
        var audioInput = context.createMediaStreamSource(stream);
        var recorder = context.createScriptProcessor(4096, channelCount, channelCount);
        var audioData = {
            size: 0,
            buffer: [],
            inputSampleRate: inputSampleRate,
            inputSampleBits: sampleBits,
            clear: function() {
                this.buffer = [];
                this.size = 0;
            },
            input: function(data) {
                this.buffer.push(new Float32Array(data));
                this.size += data.length;
            },
            encodePCM: function() {
                var bytes = new Float32Array(this.size);
                var offset = 0;
                for (var i = 0; i < this.buffer.length; i++) {
                    bytes.set(this.buffer[i], offset);
                    offset += this.buffer[i].length;
                }
                var dataLength = bytes.length * (sampleBits / 8);
                var buffer = new ArrayBuffer(dataLength);
                var data = new DataView(buffer);
                offset = 0;
                for (var i = 0; i < bytes.length; i++, offset += 2) {
                    var s = Math.max(-1, Math.min(1, bytes[i]));
                    data.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }
                return new Blob([data], { type: 'audio/pcm' });
            }
        };

        this.start = function() {
            audioInput.connect(recorder);
            recorder.connect(context.destination);
        };

        this.stop = function() {
            recorder.disconnect();
        };

        this.getBlob = function() {
            return audioData.encodePCM();
        };

        this.clear = function() {
            audioData.clear();
        };

        function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
            if (outputSampleRate === inputSampleRate) {
                return buffer;
            }
            var sampleRateRatio = inputSampleRate / outputSampleRate;
            var newLength = Math.round(buffer.length / sampleRateRatio);
            var result = new Float32Array(newLength);
            var offsetResult = 0;
            var offsetBuffer = 0;
            while (offsetResult < result.length) {
                var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
                var accum = 0, count = 0;
                for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                    accum += buffer[i];
                    count++;
                }
                result[offsetResult] = accum / count;
                offsetResult++;
                offsetBuffer = nextOffsetBuffer;
            }
            return result;
        }

        recorder.onaudioprocess = function(e) {
            // console.log('onaudioprocess called');
            var resampledData = downsampleBuffer(e.inputBuffer.getChannelData(0), inputSampleRate, outputSampleRate);
            audioData.input(resampledData);
        };
    };