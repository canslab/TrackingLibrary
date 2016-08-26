using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using Tracker;

namespace CVTest
{
    class Program
    {
        public static Rangef[] hsColorRanges = { new Rangef(0, 180), new Rangef(0, 256) };
        public static Rangef[] hColorRanges = { new Rangef(0, 180) };
        public static string roiName = "iphone.jpg";

        static void Main(string[] args)
        {
            //StopFrameTest();
            TrackerExample();
        }

        private static void TrackerExample()
        {
            ObjectTracker tracker = ObjectTracker.Instance;
            Mat modelImage = Cv2.ImRead(roiName);

            int[] channels = { 0, 1 };
            int[] binSizes = { 30, 32 };
            int dims = channels.Length;

            tracker.SetEntireArea(640, 480);
            tracker.SetModelImage(modelImage, channels, dims, binSizes, hsColorRanges);

            VideoCapture streamer = new VideoCapture(0);
            Rect meanShiftRect = new Rect();
            meanShiftRect.X = 320;
            meanShiftRect.Y = 240;

            using (Window screen = new Window("screen"))
            using (Mat eachFrame = new Mat())
            {
                while (true)
                {
                    streamer.Read(eachFrame);
                    if (eachFrame.Empty())
                    {
                        break;
                    }

                    var trackResult = tracker.TrackUsing(eachFrame, 20, meanShiftRect.X, meanShiftRect.Y);
                    meanShiftRect.X = trackResult.X;
                    meanShiftRect.Y = trackResult.Y;

                    screen.ShowImage(trackResult.Frame);

                    var key = Cv2.WaitKey(100);
                    if (key == 27)
                    {
                        Cv2.ImWrite("scan.jpg", eachFrame);
                        break;
                    }
                }
            }
        }

        private static void WellTestedHSCapturing()
        {
            VideoCapture streamer = new VideoCapture(0);

            using (Window testScreen = new Window("streamer"))
            using (Mat roiMat = Cv2.ImRead(roiName))                 // read model image
            using (Mat hsvRoiMat = roiMat.CvtColor(ColorConversionCodes.BGR2HSV))    // change format of model image
            using (Mat hsvRoiMatHisto = new Mat())
            using (Mat backProjectMat = new Mat())
            using (Mat eachFrame = new Mat()) // Frame image buffer
            {
                // 모델히스토그램 만들기
                Cv2.CalcHist(new Mat[] { hsvRoiMat }, new int[] { 0, 1 }, null, hsvRoiMatHisto, 2, new int[] { 180, 256 }, hsColorRanges);
                Cv2.Normalize(hsvRoiMatHisto, hsvRoiMatHisto, 0, 255, NormTypes.MinMax);
                Rect windowRect = new Rect(100, 100, roiMat.Width, roiMat.Height);

                while (true)
                {
                    streamer.Read(eachFrame);
                    if (eachFrame.Empty())
                    {
                        break;
                    }

                    using (Mat eachHSVFrame = eachFrame.CvtColor(ColorConversionCodes.BGR2HSV))
                    {
                        Cv2.CalcBackProject(new Mat[] { eachHSVFrame }, new int[] { 0, 1 }, hsvRoiMatHisto, backProjectMat, hsColorRanges);
                    }

                    Cv2.MeanShift(backProjectMat, ref windowRect, TermCriteria.Both(20, 1));
                    Cv2.Rectangle(eachFrame, windowRect, 255, 3);

                    testScreen.ShowImage(eachFrame);
                    //testScreen.ShowImage(backProjectMat.CvtColor(ColorConversionCodes.GRAY2BGR));
                    //testScreen.ShowImage(backProjectMat);
                    var key = Cv2.WaitKey(100);

                    if (key == 27)
                    {
                        break;
                    }
                }
            }
        }

        private static void Testing1()
        {

            var frame = Cv2.ImRead("frame.jpg");
            var hsvFrame = frame.CvtColor(ColorConversionCodes.BGR2HSV);

            var roi = Cv2.ImRead("redBox.jpg");
            var hsvRoi = roi.CvtColor(ColorConversionCodes.BGR2HSV);
            var hsvRoiHist = new Mat();


            Cv2.CalcHist(new Mat[] { hsvRoi }, new int[] { 0, 1 }, null, hsvRoiHist, 2, new int[] { 180, 256 }, hsColorRanges);
            Cv2.Normalize(hsvRoiHist, hsvRoiHist, 0, 255, NormTypes.MinMax);

            var backProjectMat = new Mat();

            Cv2.CalcBackProject(new Mat[] { hsvFrame }, new int[] { 0, 1 }, hsvRoiHist, backProjectMat, hsColorRanges);
            Cv2.ImShow("backproject image", backProjectMat);
            Cv2.WaitKey(0);
        }

        private static void StopFrameTest()
        {
            ObjectTracker tracker = ObjectTracker.Instance;
            Mat modelImage = Cv2.ImRead("target.jpg");
            Mat scanImage = Cv2.ImRead("scan.jpg");
            int[] channels = { 0, 1 };
            int[] binSizes = { 30, 32 };
            int dims = channels.Length;

            tracker.SetEntireArea(640, 480);
            tracker.SetModelImage(modelImage, channels, dims, binSizes, hsColorRanges);
            var result = tracker.TrackUsing(scanImage, 20, 320,240);

            Cv2.ImShow("result", result.Frame);
            Cv2.WaitKey(0);

        }
    }
}
