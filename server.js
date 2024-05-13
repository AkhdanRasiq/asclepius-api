require('dotenv').config();

const Hapi = require('@hapi/hapi');
const Joi = require('joi');
const admin = require('firebase-admin');
const tf = require('@tensorflow/tfjs-node');
const serviceAccount = require('./dicoding-ml.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

async function predictClassification(model, image) {
    const tensor = tf.node
        .decodeJpeg(image)
        .resizeNearestNeighbor([224, 224])
        .expandDims()
        .toFloat()
    
    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const classes = ['Cancer', 'Non Cancer'];

    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    const label = classes[classResult];

    let suggestion;
 
    if (label === 'Cancer') {
        suggestion = "Segera periksa ke dokter!"
    }
    
    if (label === 'Non Cancer') {
        suggestion = "Tidak perlu periksa ke dokter!"
    }

    return { confidenceScore, label, suggestion };
}

async function postPredictHandler(request, predictionID) {
    const { image } = request.payload;
    const { model } = request.server.app;

    const { confidenceScore, label, suggestion } = await predictClassification(model, image);

    const createdAt = new Date().toISOString();

    const response = {
        status: 'success',
        message: confidenceScore > 99 ? 'Model is predicted successfully.' : 'Model is predicted successfully but under threshold. Please use the correct picture',
        data: {
            id: predictionID,
            result: label,
            suggestion: suggestion,
            createdAt: createdAt
        }
      }

      // Save response data to Firestore
      const docRef = await db.collection('predictions').doc(predictionID);
      await docRef.set(response.data);

      return response;
}

async function loadModel() {
    return tf.loadGraphModel(process.env.MODEL_URL);
}

const init = async () => {
    const server = Hapi.server({
        port: 3000,
        host: 'localhost',
        routes: {
            cors: {
                origin: ['*'],
              },
        }
    });

    const model = await loadModel();
    server.app.model = model;

    // Route for prediction
    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                allow: 'multipart/form-data',
                multipart: true,
                maxBytes: 1000000
            },
            validate: {
                    payload: Joi.object({
                        image: Joi.any().required().meta({ swaggerType: 'file' })
                    }).unknown()
                }
        },
        handler: async (request, h) => {
            try {
                const { image } = request.payload;

                // Check if image size exceeds maximum allowed size
                if (image.length > 1000000) {
                    return h.response({
                        status: 'fail',
                        message: 'Payload content length greater than maximum allowed: 1000000'
                    }).code(413);
                }

                // Prepare response
                const predictionID = crypto.randomUUID();
                const response = await postPredictHandler(request, predictionID);

                return h.response(response).code(201);
            } catch (error) {
                console.error(error);
                return h.response({
                    status: 'fail',
                    message: 'Terjadi kesalahan dalam melakukan prediksi'
                }).code(400);
            }
        }
    });

    await server.start();
    console.log('Server running on %s', server.info.uri);
};

process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

init();
