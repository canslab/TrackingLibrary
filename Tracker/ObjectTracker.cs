using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using System.Diagnostics;
using System.Collections.Concurrent;

namespace Tracker
{
    public class ObjectTracker
    {
        // static variables 
        public static readonly Rangef[] HueSatColorRanges = { new Rangef(0, 180), new Rangef(0, 256) };
        public static ObjectTracker Instance { get; }

        public Point mInitialLocation = new Point(0, 0);

        public int FindAreaWidth { get; private set; }
        public int FindAreaHeight { get; private set; }

        public Mat CvtModelImage { get; private set; } = null;
        public Mat ModelHistogram { get; private set; } = null;
        private bool IsFindAreaSettingDone { get; set; } = false;
        private bool IsModelHistogramReady { get; set; } = false;

        public int[] Channels { get; private set; } = null;
        public int Dims { get; private set; } = 0;
        public int[] HistSize { get; private set; } = null;
        public Rangef[] ColorRanges { get; private set; } = null;

        public bool IsTrackingReady
        {
            get
            {
                return (IsFindAreaSettingDone && IsModelHistogramReady);
            }
        }
        static ObjectTracker()
        {
#if SHOW_LOG
            Console.WriteLine("ObjectTracker 생성");
#endif
            Instance = new ObjectTracker();
        }
        public void SetEntireArea(int width, int height)
        {
            Debug.Assert(width > 0 && height > 0);
            FindAreaWidth = width;
            FindAreaHeight = height;

            IsFindAreaSettingDone = true;
        }

        public void SetModelImageAndMakeHistogram(Mat modelImage, int[] channels, int dims, int[] histSize, Rangef[] colorRanges)
        {
            Debug.Assert(modelImage != null && modelImage.IsDisposed == false);
            Debug.Assert(channels != null && channels.Length <= 3 && dims <= 3);
            Debug.Assert(histSize != null && histSize.Length == dims);
            Debug.Assert(colorRanges != null && colorRanges.Length == dims);
            Debug.Assert(ModelHistogram == null);
            Debug.Assert(IsFindAreaSettingDone == true);

            // 모델 히스토그램 행렬 생성
            ModelHistogram = new Mat();

            // 모델이미지를 HSV 포맷으로 변환 -> 히스토그램 계산 -> 노멀라이즈
            CvtModelImage = modelImage.CvtColor(ColorConversionCodes.BGR2HSV);
            Cv2.CalcHist(new Mat[] { CvtModelImage }, channels, null, ModelHistogram, dims, histSize, colorRanges);
            Cv2.Normalize(ModelHistogram, ModelHistogram, 0, 255, NormTypes.MinMax);

            // 채널, 히스토그램 사이즈, 컬러범위 깊은 복사
            Channels = new int[channels.Length];
            HistSize = new int[histSize.Length];

            // 차원 복사
            Dims = dims;
            for (int i = 0; i < channels.Length; ++i)
            {
                Channels[i] = channels[i];
                HistSize[i] = histSize[i];
            }
            ColorRanges = colorRanges;

            IsModelHistogramReady = true;
        }
        public TrackResult TrackUsing(Mat currentFrame, int nThread, int hintX, int hintY)
        {
            Debug.Assert(IsTrackingReady == true);
            Debug.Assert(currentFrame != null && currentFrame.IsDisposed == false);
            Debug.Assert(Channels != null && Dims > 0 && HistSize != null && ColorRanges != null);
            Debug.Assert(nThread >= 1);

            // 트랙킹 결과
            TrackResult result = null;
            Random randomGenerator = new Random();

            // nThread 수만 큼 윈도우 생성, (병렬로 mean shift를 돌리기 위함)
            Rect[] meanShiftRects = new Rect[nThread];

            // 유저에게 입력받은 힌트는 후보군에 하나 넣어둔다.
            meanShiftRects[0].X = hintX;
            meanShiftRects[0].Y = hintY;
            meanShiftRects[0].Width = CvtModelImage.Width;
            meanShiftRects[0].Height = CvtModelImage.Height;

            GenerateRandomRects(randomGenerator, meanShiftRects, 1);

            using (Mat backProjectMat = new Mat())
            {
                // 백프로젝션 하기
                using (Mat cvtCurrentFrame = currentFrame.CvtColor(ColorConversionCodes.BGR2HSV))
                {
                    Cv2.CalcBackProject(new Mat[] { cvtCurrentFrame }, Channels, ModelHistogram, backProjectMat, ColorRanges);
                }

                // 병렬로 mean shift 
                var resultRect = ParallelMeanShift(currentFrame, meanShiftRects, backProjectMat);
                Cv2.Rectangle(currentFrame, resultRect, new Scalar(255, 255, 0), 3);

                result = new TrackResult(resultRect.X, resultRect.Y, currentFrame);
            }

            return result;
        }

        private Rect ParallelMeanShift(Mat currentFrame, Rect[] rects, Mat backProjectMat)
        {
            ConcurrentDictionary<int, double> similarityDic = new ConcurrentDictionary<int, double>(); 

            // nThread(rect의 갯수)만큼 Parallel하게 mean shift 수행
            Parallel.For(0, rects.Length, (index) =>
            {
                Cv2.MeanShift(backProjectMat, ref rects[index], TermCriteria.Both(30, 1));
                similarityDic[index] = Cv2.Mean(backProjectMat[rects[index]]).Val0;
#if SHOW_LOG
                lock (currentFrame)
                {
                    Cv2.Rectangle(currentFrame, rects[index], 255, 3);
                }
                Console.WriteLine(Cv2.Mean(backProjectMat.SubMat(rects[index])));
#endif
            });

            // get top 5! 
            var sortedKeys = (from number in similarityDic.Keys
                              orderby similarityDic[number] descending
                              select number).ToArray();

#if SHOW_LOG
            for(int i = 0; i <= rects.Length / 5; ++i)
            {
                Console.Write(similarityDic[sortedKeys[i]]+" ");
            }
            Console.WriteLine("\n--------------");
#endif
            return rects[sortedKeys[0]];
        }

        private void GenerateRandomRects(Random randomGenerator, Rect[] rects, int from)
        {
            for (int index = from; index < rects.Length; ++index)
            {
                // 각 rect별로 랜덤하게 포인트를 잡는다.
                rects[index].X = randomGenerator.Next() % (640 - CvtModelImage.Width);
                rects[index].Y = randomGenerator.Next() % (480 - CvtModelImage.Height);
                rects[index].Width = CvtModelImage.Width;
                rects[index].Height = CvtModelImage.Height;
            }
        }

        public class TrackResult : IDisposable
        {
            public int X { get; private set; }
            public int Y { get; private set; }
            public int CenterY { get; }
            public Mat Frame { get; private set; }
            public TrackResult(int x, int y, Mat frame)
            {
                Debug.Assert(x >= 0 && y >= 0 && frame != null);
                X = x;
                Y = y;
                Frame = frame.Clone();
            }

            public void Dispose()
            {
                ResetAllResources();
                GC.SuppressFinalize(this);
                IsDisposed = true;
            }

            private void ResetAllResources()
            {
                Frame?.Release();
                Frame = null;
                X = 0;
                Y = 0;
            }

            ~TrackResult()
            {
                if (IsDisposed == false)
                {
                    ResetAllResources();
                }
            }
            private bool IsDisposed { get; set; } = false;
        }

    }
}
