using System;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Threading;

namespace Tracker
{
    public class ObjectTracker
    {
        // User에게 제공해주는 Rangef[] 상수들 
        public static readonly Rangef[] HueSatColorRanges = { new Rangef(0, 180), new Rangef(0, 256) };
        public static readonly Rangef[] HueColorRanges = { new Rangef(0, 180) };

        /// <summary>
        /// 싱글턴 객체를 얻어온다.
        /// </summary>
        public static ObjectTracker Instance { get; }

        public int FindAreaWidth { get; private set; }
        public int FindAreaHeight { get; private set; }

        public Mat CvtModelImage { get; private set; }
        public Mat ModelHistogram { get; private set; }
        private bool IsFindAreaSettingDone { get; set; }
        private bool IsModelHistogramReady { get; set; }
        public bool IsTrackingReady
        {
            get
            {
                return (IsFindAreaSettingDone && IsModelHistogramReady);
            }
        }

        public int[] Channels { get; private set; }
        public int Dims { get; private set; }
        public int[] HistSize { get; private set; }
        public Rangef[] ColorRanges { get; private set; }

        static ObjectTracker()
        {
#if SHOW_LOG
            Console.WriteLine("ObjectTracker 생성");
#endif
            Instance = new ObjectTracker();
        }

        private ObjectTracker()
        {
            InitPropertiesAndVariables();
        }

        /// <summary>
        /// 프레임의 크기.
        /// </summary>
        /// <param name="width"> 프레임 너비 </param>
        /// <param name="height"> 프레임 높이 </param>
        public void SetEntireArea(int width, int height)
        {
            Debug.Assert(width > 0 && height > 0);
            FindAreaWidth = width;
            FindAreaHeight = height;

            IsFindAreaSettingDone = true;
        }

        /// <summary>
        /// 모델 이미지에 대한 히스토그램을 만들어서 내부에 저장해둔다.
        /// </summary>
        /// <param name="modelImage"> 모델 이미지 </param>
        /// <param name="channels"> 채널 수 </param>
        /// <param name="dims"> 차원 수 </param>
        /// <param name="histSize"> 히스토그램 사이즈 </param>
        /// <param name="colorRanges"> 컬러 범위 </param>
        public void SetModelImage(Mat modelImage, int[] channels, int dims, int[] histSize, Rangef[] colorRanges)
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

        /// <summary>
        /// currentFrame을 가지고서 Tracking을 한다, 그 결과 TrackResult를 반환한다.
        /// 그 결과를 사용한 뒤 Dispose하는것은 Caller의 책임이다.
        /// </summary>
        /// <param name="currentFrame"> 현재 프레임 </param>
        /// <param name="nThread"> 병렬성의 정도, 높을 수록 부하가 높으나, 트랙킹 결과는 좋아짐.</param>
        /// <param name="hintX"> 힌트가 되는 초기 x좌표 </param>
        /// <param name="hintY"> 힌트가 되는 초기 y좌표 </param>
        /// <returns></returns>
        public TrackResult TrackUsing(Mat currentFrame, int nThread, int hintX, int hintY)
        {
            Debug.Assert(IsTrackingReady == true);
            Debug.Assert(currentFrame != null && currentFrame.IsDisposed == false);
            Debug.Assert(Channels != null && Dims > 0 && HistSize != null && ColorRanges != null);
            Debug.Assert(nThread >= 1);

            // 트랙킹 결과
            TrackResult retResult = null;
            Random randomGenerator = new Random();

            // nThread 수만 큼 윈도우 생성, (병렬로 mean shift를 돌리기 위함)
            Rect[] candidateRects = new Rect[nThread];

            // 유저에게 입력받은 힌트는 후보군에 하나 넣어둔다.
            candidateRects[0].X = hintX;
            candidateRects[0].Y = hintY;
            candidateRects[0].Width = CvtModelImage.Width;
            candidateRects[0].Height = CvtModelImage.Height;

            GenerateRandomRects(randomGenerator, candidateRects, 1);

            using (Mat backProjectMat = new Mat())
            {
                // 백프로젝션 하기
                using (Mat cvtCurrentFrame = currentFrame.CvtColor(ColorConversionCodes.BGR2HSV))
                {
                    Cv2.CalcBackProject(new Mat[] { cvtCurrentFrame }, Channels, ModelHistogram, backProjectMat, ColorRanges);
                }

                // 병렬로 mean shift 
                retResult = ParallelMeanShift(currentFrame, candidateRects, backProjectMat);
                Cv2.Rectangle(currentFrame, new Rect(retResult.X, retResult.Y, retResult.Width, retResult.Height), new Scalar(255, 255, 0), 3);
            }

            return retResult;
        }

        /// <summary>
        /// 트랙커를 reset한다.(내부 리소스 모드 해제 및 초기화)
        /// </summary>
        public void ResetTrackerAndRelease()
        {
            InitPropertiesAndVariables();
        }

        private void InitPropertiesAndVariables()
        {
            // 이미지들 release
            CvtModelImage?.Release();
            ModelHistogram?.Release();

            FindAreaHeight = 0;
            FindAreaWidth = 0;

            Channels = null;
            Dims = 0;
            HistSize = null;
            ColorRanges = null;

            IsFindAreaSettingDone = false;
            IsModelHistogramReady = false;

            CvtModelImage = null;
            ModelHistogram = null;
        }

        private TrackResult ParallelMeanShift(Mat currentFrame, Rect[] candidateRects, Mat backProjectMat)
        {
            ConcurrentDictionary<int, double> similarityDic = new ConcurrentDictionary<int, double>();
            TrackResult retResult = null;

            // nThread(rect의 갯수)만큼 Parallel하게 mean shift 수행
            Parallel.For(0, candidateRects.Length, (index) =>
            {
                Cv2.MeanShift(backProjectMat, ref candidateRects[index], TermCriteria.Both(30, 1));
                similarityDic[index] = Cv2.Mean(backProjectMat[candidateRects[index]]).Val0;
#if SHOW_LOG
                lock (currentFrame)
                {
                    Cv2.Rectangle(currentFrame, candidateRects[index], 255, 3);
                }
               
                Console.WriteLine("tid = {0}, mean value = {1}", Thread.CurrentThread.ManagedThreadId, similarityDic[index]);
#endif
            });

            // get top 5! 
            var sortedKeys = (from number in similarityDic.Keys
                              orderby similarityDic[number] descending
                              select number).ToArray();

            var largestValue = similarityDic[sortedKeys[0]];
            var smallestValue = similarityDic[sortedKeys[candidateRects.Length - 1]];

            var stdev = MyMath.GetStdev(similarityDic.Values.ToArray());

            // 표준편차가 매우 작다는 이야기는, object가 없음을 의미
            if (stdev < 5)
            {
                retResult = new TrackResult(0, 0, 0, 0, false);
                Console.WriteLine("타깃없음");
            }
            else
            {
                retResult = new TrackResult(candidateRects[sortedKeys[0]], true);
                Console.WriteLine("x = {0}, y = {1}, width = {2}, height = {3}", retResult.X, retResult.Y, retResult.Width, retResult.Height);

            }
#if SHOW_LOG
            // 상위 10% 표시
            for(int i = 0; i <= candidateRects.Length / 5; ++i)
            {
                Console.Write(similarityDic[sortedKeys[i]]+" ");
            }
            Console.WriteLine("젤 큰놈 = {0}, 젤 작은 놈 = {1}", largestValue, smallestValue);
            Console.WriteLine("표준편차 = {0}", stdev);
            Console.WriteLine("\n--------------");
#endif
            return retResult;
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
            public Rect Region
            {
                get
                {
                    return new Rect(X, Y, Width, Height);
                }
            }
            public int X { get; private set; }
            public int Y { get; private set; }

            public int Width { get; private set; }
            public int Height { get; private set; }

            public bool IsObjectExist { get; private set; }

            public TrackResult(int x, int y, int width, int height, bool bTrackedWell)
            {
                Debug.Assert(x >= 0 && y >= 0);
                X = x;
                Y = y;
                Width = width;
                Height = height;
                IsObjectExist = bTrackedWell;
            }

            public TrackResult(Rect rect, bool bTrackedWell)
            {
                X = rect.X;
                Y = rect.Y;
                Width = rect.Width;
                Height = rect.Height;

                IsObjectExist = bTrackedWell;
            }

            public void Dispose()
            {
                ResetAllResources();
                GC.SuppressFinalize(this);
                IsDisposed = true;
            }

            private void ResetAllResources()
            {
                X = 0;
                Y = 0;
                Width = 0;
                Height = 0;
                IsObjectExist = false;
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
