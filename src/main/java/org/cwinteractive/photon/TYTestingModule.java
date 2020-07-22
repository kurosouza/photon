package org.cwinteractive.photon;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.SvhnLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import picocli.CommandLine;

@CommandLine.Command(name = "test", description = "Test model")
public class TYTestingModule implements Runnable {

    private static final Logger log = LoggerFactory.getLogger(TYTestingModule.class);

    // Enable different colour bounding box for different classes
    public static final Scalar RED = RGB(255.0, 0, 0);
    public static final Scalar GREEN = RGB(0, 255.0, 0);
    public static final Scalar BLUE = RGB(0, 0, 255.0);
    public static final Scalar YELLOW = RGB(255.0, 255.0, 0);
    public static final Scalar CYAN = RGB(0, 255.0, 255.0);
    public static final Scalar MAGENTA = RGB(255.0, 0.0, 255.0);
    public static final Scalar ORANGE = RGB(255.0, 128.0, 0);
    public static final Scalar PINK = RGB(255.0, 192.0, 203.0);
    public static final Scalar LIGHTBLUE = RGB(153.0, 204.0, 255.0);
    public static final Scalar VIOLET = RGB(238.0, 130.0, 238.0);

    public void run() {
        try {
            int width = 416;
            int height = 416;
            int nChannels = 3;
            int gridWidth = 13;
            int gridHeight = 13;

            // number classes (digits) for the SVHN datasets
            int nClasses = 10;

            int nBoxes = 5;
            double lambdaNoObj = 0.5;
            double lambdaCoord = 1.0;
            double[][] priorBoxes = { { 2, 5 }, { 2.5, 6 }, { 3, 7 }, { 3.5, 8 }, { 4, 9 } };
            double detectionThreshold = 0.5;

            // parameters for the training phase
            int batchSize = 10;
            int nEpochs = 20;
            double learningRate = 1e-4;

            int seed = 123;
            Random rng = new Random(seed);

            SvhnDataFetcher fetcher = new SvhnDataFetcher();
            File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
            File testDir = fetcher.getDataSetPath(DataSetType.TEST);

            log.info("Load data...");

            FileSplit trainData = new FileSplit(trainDir,
            NativeImageLoader.ALLOWED_FORMATS, rng);
            FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

            ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                    gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
            recordReaderTrain.initialize(trainData);

            ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                    gridHeight, gridWidth, new SvhnLabelProvider(testDir));
            recordReaderTest.initialize(testData);

            // ObjectDetectionRecordReader performs regression, so we need to specify it
            // here

            RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1,
                    true);
            train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

            RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
            test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

            ComputationGraph model;
            String modelFilename = "model.zip";

            if (new File(modelFilename).exists()) {
                log.info("Load model...");

                model = ComputationGraph.load(new File(modelFilename), true);

                // visualize results on the test set
                NativeImageLoader imageLoader = new NativeImageLoader();
                CanvasFrame frame = new CanvasFrame("HouseNumberDetection");
                OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
                org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model
                        .getOutputLayer(0);
                List<String> labels = train.getLabels();
                test.setCollectMetaData(true);
                Scalar[] colormap = { RED, BLUE, GREEN, CYAN, YELLOW, MAGENTA, ORANGE, PINK, LIGHTBLUE, VIOLET };

                while (test.hasNext() && frame.isVisible()) {
                    org.nd4j.linalg.dataset.DataSet ds = test.next();
                    RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get(0);
                    INDArray features = ds.getFeatures();
                    INDArray results = model.outputSingle(features);
                    List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
                    File file = new File(metadata.getURI());
                    log.info(file.getName() + ": " + objs);

                    Mat mat = imageLoader.asMat(features);
                    Mat convertedMat = new Mat();
                    mat.convertTo(convertedMat, CV_8U, 255, 0);
                    int w = metadata.getOrigW() * 2;
                    int h = metadata.getOrigH() * 2;
                    Mat image = new Mat();
                    resize(convertedMat, image, new Size(w, h));
                    for (DetectedObject obj : objs) {
                        double[] xy1 = obj.getTopLeftXY();
                        double[] xy2 = obj.getBottomRightXY();
                        String label = labels.get(obj.getPredictedClass());
                        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                        int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                        int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                        rectangle(image, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()]);
                        putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1,
                                colormap[obj.getPredictedClass()]);

                    }
                    frame.setTitle(new File(metadata.getURI()).getName() + " - HouseNumberDetection");
                    frame.setCanvasSize(w, h);
                    frame.showImage(converter.convert(image));
                    frame.waitKey();
                }
                frame.dispose();

            }

        } catch (IOException ioe) {
            log.error("IOException: ", ioe);
        } catch (InterruptedException ie) {
            log.error("InterruptedException: ", ie);
        }

    }

}