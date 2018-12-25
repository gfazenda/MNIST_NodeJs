import * as tf from '@tensorflow/tfjs';
import {MnistData} from './data.js';

var model, data;



const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);

export class TfModel {
    constructor() {
      this.shuffledTrainIndex = 0;
      this.shuffledTestIndex = 0;
    }

    async createModelAndTrain(){
        this.createModel();
        console.log('loading');
        await this.loadData();
        console.log('training');
        this.train();
        console.log('trained');
    }

 createModel() {
    model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));


    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }));



    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    console.log('model created');
}

async  loadData(){
    data = new MnistData();
    await data.load();
}

async  train() {
    console.log('Start training ...');
    for (let i = 0; i < TRAIN_BATCHES; i++) {
        const batch = tf.tidy(() => {
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;
        });

        await model.fit(
            batch.xs, batch.labels, {batchSize: BATCH_SIZE, epochs: 1}
        );

        tf.dispose(batch);

        await tf.nextFrame();
    }
    console.log('Training complete');
}

async  predict(batch) {
    tf.tidy(() => {
        const input_value = Array.from(batch.labels.argMax(1).dataSync());

        const div = document.createElement('div');
        div.className = 'prediction-div';

        const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

        const prediction_value = Array.from(output.argMax(1).dataSync());
        const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]);

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
       // draw(image.flatten(), canvas);

        const label = document.createElement('div');
        label.innerHTML = 'Original Value: ' + input_value;
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;

        if (prediction_value - input_value == 0) {
            label.innerHTML += '<br>Value recognized successfully';
        } else {
            label.innerHTML += '<br>Recognition failed!'
        }

        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('predictionResult').appendChild(div);
    });
}

 draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}
}