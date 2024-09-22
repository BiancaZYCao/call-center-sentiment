var recordButton = document.getElementById('recordButton');
var transcriptionResult = document.getElementById('transcriptionResult');
var sentimentResult = document.getElementById('sentimentResult');
// add by me 20240903
var responseResult = document.getElementById('responseResult');
//end of code 20240903
var topicModelResult = document.getElementById('topicModelResult');

var singlishResult = document.getElementById('SinglishResult');



var pendingTextData = '';    // 存储需要合并的短 textData
var minWordCount = 7;        // 定义最少的词数


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
    if (!isRecording) {  //麦克风当前没有在录音，应该开始录音
        startListening();
        startRecording();
    } else {  //麦克风已经在录音，应该停止录音
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
        queryParams.push(`lang=${lang}`);  // 如果指定了语言，添加到查询参数
    }
    if (sv) {
        queryParams.push('sv=1');  // 如果启用了说话人验证，添加到查询参数
    }
    var queryString = queryParams.length > 0 ? `?${queryParams.join('&')}` : '';   // 拼接查询字符串

    //1. 建立 WebSocket 连接
    ws = new WebSocket(`ws://127.0.0.1:8000/ws/transcribe${queryString}`);  // 创建WebSocket连接，使用查询字符串
    ws.binaryType = 'arraybuffer';   // 设置WebSocket传输的二进制数据类型
    //2. 处理连接成功的事件
    ws.onopen = function (event) {
        console.log('WebSocket connection established');
        //2.a 启动录音
        record.start();  // record是一个全局变量，用于存储录音实例
        // setInterval()是一个浏览器和 Node.js 环境中常用的定时器函数，它会在指定的时间间隔内反复执行某个代码块
        timeInte = setInterval(function () {
            if (ws.readyState === 1) {  // 如果WebSocket连接是打开的状态
                var audioBlob = record.getBlob();   // 获取录音的Blob对象(JavaScript 中的一种对象，用于存储二进制数据。它可以存储各种类型的文件数据，比如图片、音频、视频)
                // console.log('Blob size: ', audioBlob.size);

                // Read the Blob content for debugging
                var reader = new FileReader();
                reader.onloadend = function () {
                    // console.log('Blob content: ', new Uint8Array(reader.result));
                    //2.b 发送音频数据到服务器
                    ws.send(audioBlob);
                    console.log('Sending audio data');
                    record.clear(); // 清除录音缓冲区
                };
                reader.readAsArrayBuffer(audioBlob); // 将Blob对象读取为ArrayBuffer
            }
        }, 500);  // 每500毫秒发送一次音频数据
    };
    //3. 处理服务器发送的消息
    // ws.onmessage:当websocket server接收到消息触发，去处理从服务器返回的数据，并更新用户界面或执行其他任务
    ws.onmessage = function (evt) {
        console.log('Received message: ' + evt.data);
        try {
            var resJson = JSON.parse(evt.data);  // 尝试解析JSON数据
            var textData = resJson.data;         // 提取转录的文本数据
            var speaker = resJson.speaker_label || 'unknown speaker'; // Handle missing speaker_label
            var type = resJson.type || 'unknown type';
            var timestamp = resJson.timestamp || 'no timestamp';


            // 显示情感分析结果 音频情感分析处理 - new model- 20240915
            if (type === 'audio_sentiment') {
                var audioData = JSON.parse(textData);  // Parse the JSON string in 'data'
                var finalScore = Number(audioData.final_score);  // Cast final_score to a number
                updateGauge(finalScore, audioData.final_sentiment_3);  // Update gauge
                responseResult.textContent = `Final Sentiment (3 Classes): ${audioData.final_sentiment_3}\n`;
                startFetchChart = true;
            }

            // 显示转录结果 加上说话者身份
            if (type === 'STT') {
                if (speaker === speaker_last) {
                    transcriptionResult.innerHTML += ' ' + textData || ' ';
                } else {
                    speaker_last = speaker;
                    transcriptionResult.innerHTML += "<br><strong>" + speaker + ":</strong> " + textData;
                }
            }
        } catch (err) {
            console.error('Failed to parse websocket message:', err);
            transcriptionResult.textContent += "\n" + evt.data;  // 如果解析失败，直接显示原始数据
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
                    method: 'POST',  // Use POST method to send the request
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
                        tagsContainer.removeChild(tagElement);  // Remove the tag when the close button is clicked
                    };

                    tagElement.appendChild(closeButton);
                    tagsContainer.appendChild(tagElement);
                });
            } else {
                // Display an error message if no topics are found
                const topicModelResult = document.getElementById('topicModelResult');
                // topicModelResult.textContent = `Error: No topics found or ${resJson.error}`;  // Display error information
            }
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
            init(new Recorder(mediaStream));    // 成功获取音频输入时，初始化Recorder对象
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