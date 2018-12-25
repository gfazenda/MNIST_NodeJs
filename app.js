var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var multer  = require('multer');
var fs = require("fs");
var jpeg = require('jpeg-js');

var upload = multer({ dest: '/tmp/'});
var app = express();
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const data = require('./src/data');
const model = require('./src/model');



// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

const imageHeight=28,imageWidth=28;

var port = process.env.PORT || 8000;
var modelCreated = false;
app.listen(port, function(){
	console.log('run port: ' + port);
});


app.get('/', function(req, res){
 if(modelCreated==false){
  console.log('the model has been created and is being trained')
  var createAndTrainModel = startup(1,128);
  createAndTrainModel.then(function(){
    modelCreated=true;
    console.log('training completed');
    res.render('index.pug');
  });
 }else{
  res.render('index.pug');
 }
 
});


// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

app.post('/test', function(req, res) {
  console.log('ooloko');
 // main.execute();
 // res.render('error');
});

const handleError = (err, res) => {
  console.log(err);
  res
    .status(500)
    .contentType("text/plain")
    .end("didnt work");
};

app.post(
  "/newImage",
  upload.single("file"),
  (req, res) => {
    const tempPath = req.file.path;
    const targetPath = path.join(__dirname, "./public/images/image.jpg");
    console.log(targetPath);
    console.log(req.file.originalname);
    if (path.extname(req.file.originalname).toLowerCase() === ".jpg") {
      fs.rename(tempPath, targetPath, err => {
        if (err) return handleError(err, res);

        let number = doPrediction(targetPath);
        res.render('predict.pug',{ result: number });
      });
    } else {
      fs.unlink(tempPath, err => {
        if (err) return handleError(err, res);

        res
          .status(403)
          .contentType("text/plain")
          .end("Only .jpg files are allowed!");
      });
    }
  }
);

function readImage(path){
  const buf = fs.readFileSync(path);
  const pixels = jpeg.decode(buf, true);
  return pixels
}

function convertGreyScale(image){
  let pixels = imageHeight*imageWidth;
  const array = new Int32Array(pixels);
  let pIndex = 0;
  for (let index = 0; index < image.data.length; index+=4) {
    array[pIndex] = image.data[index];
    pIndex++;
  }
  console.log(array.length);
  return array;
}

function doPrediction(path){
  var imgData = convertGreyScale(readImage(path)); 
  let input = tf.tensor(imgData).reshape([1,28,28,1]).cast('float32').div(tf.scalar(255)); //transform to tensor4d
  console.log(input);
  const predict = model.predict(input).dataSync();
  console.log(getResult(predict));
  return getResult(predict);
}

function getResult(array){ //returns the index of the value predicted by the model
  if (array.length === 0) {
      return -1;
  }
  let max = array[0];
  let maxIndex = 0;
  for (let i = 1; i < array.length; i++) {
      if (array[i] > max) {
          maxIndex = i;
          max = array[i];
      }
  }
  return maxIndex;
}

async function startup(epochs, batchSize, modelSavePath) {
    await data.loadData();
  
    const {images: trainImages, labels: trainLabels} = data.getTrainData();
    model.summary();
  
    let epochBeginTime;
    let millisPerStep;
    const validationSplit = 0.15;
    const numTrainExamplesPerEpoch =
        trainImages.shape[0] * (1 - validationSplit);
    const numTrainBatchesPerEpoch =
        Math.ceil(numTrainExamplesPerEpoch / batchSize);

    console.log('fitting')
    console.log(numTrainExamplesPerEpoch)
    console.log(numTrainBatchesPerEpoch)
    await model.fit(trainImages, trainLabels, {
      epochs,
      batchSize,
      validationSplit
    });
    console.log('fitted')
    const {images: testImages, labels: testLabels} = data.getTestData();
    const evalOutput = model.evaluate(testImages, testLabels);
  
    console.log(
        `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
}  

module.exports = app;
