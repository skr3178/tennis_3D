using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace TennisDataCollection.Runtime
{
    // Attached to the Ball GameObject. Records per-FixedUpdate state during Play Mode.
    // The existing tennis game's Ball / Bot / Player scripts are untouched — this is
    // an observer, not a modifier.
    public class TrajectoryRecorder : MonoBehaviour
    {
        Rigidbody body;
        Camera mainCam;

        // Output image resolution used both for the (u,v) projection and exported intrinsics.
        // In batchmode -nographics there's no real Game window, so Camera.pixelWidth is unreliable.
        public int outWidth = 1280;
        public int outHeight = 720;

        // Subtracted from world-y when saving so the court surface ends up at y=0
        // (paper convention). The same offset is also subtracted from the saved
        // camera position so reprojection of (x, y_saved, z) using the saved
        // camera remains pixel-identical to the in-engine recording.
        public float groundOffsetY = 0f;

        struct FrameRec
        {
            public int frame;
            public float t;
            public Vector3 pos;
            public Vector3 vel;
            public Vector2 pixel;
            public bool eot;
        }

        struct CameraSnapshot
        {
            public float fx, fy, cx, cy;
            public int width, height;
            public float fovYDeg;
            public Matrix4x4 worldToCamera;   // Unity's view matrix (OpenGL convention: cam looks down -Z)
            public Matrix4x4 cameraToWorld;
            public Vector3 position;
            public Quaternion rotation;
        }

        readonly List<FrameRec> frames = new List<FrameRec>();
        CameraSnapshot camSnap;
        bool recording;
        float startTime;
        int frameIdx;
        bool pendingEoT;

        public int FrameCount => frames.Count;
        public bool IsRecording => recording;

        void Awake()
        {
            body = GetComponent<Rigidbody>();
            mainCam = Camera.main;
        }

        public void SetCamera(Camera c) => mainCam = c;
        public void SetOutputResolution(int w, int h) { outWidth = w; outHeight = h; }
        public void SetGroundOffsetY(float y) { groundOffsetY = y; }

        public void BeginRecording()
        {
            frames.Clear();
            frameIdx = 0;
            pendingEoT = false;
            startTime = Time.time;
            CaptureCameraSnapshot();
            recording = true;
        }

        void CaptureCameraSnapshot()
        {
            camSnap = new CameraSnapshot();
            if (mainCam == null) return;
            camSnap.width = outWidth;
            camSnap.height = outHeight;
            camSnap.fovYDeg = mainCam.fieldOfView;
            float fovY = mainCam.fieldOfView * Mathf.Deg2Rad;
            camSnap.fy = (outHeight * 0.5f) / Mathf.Tan(fovY * 0.5f);
            camSnap.fx = camSnap.fy;                       // square pixels (Unity uses vertical FOV)
            camSnap.cx = outWidth  * 0.5f;
            camSnap.cy = outHeight * 0.5f;
            camSnap.worldToCamera = mainCam.worldToCameraMatrix;
            camSnap.cameraToWorld = mainCam.cameraToWorldMatrix;
            camSnap.position = mainCam.transform.position;
            camSnap.rotation = mainCam.transform.rotation;
        }

        public void FlagHitOnPreviousFrame()
        {
            if (!recording) return;
            pendingEoT = true;
        }

        public int EndRecording(bool markLastAsEoT = true)
        {
            recording = false;
            if (markLastAsEoT && frames.Count > 0)
            {
                var last = frames[frames.Count - 1];
                last.eot = true;
                frames[frames.Count - 1] = last;
            }
            return frames.Count;
        }

        void FixedUpdate()
        {
            if (!recording) return;

            if (pendingEoT && frames.Count > 0)
            {
                var prev = frames[frames.Count - 1];
                prev.eot = true;
                frames[frames.Count - 1] = prev;
                pendingEoT = false;
            }

            Vector3 pos = transform.position;
            Vector3 vel = body != null ? body.linearVelocity : Vector3.zero;
            Vector2 px = ProjectToPixel(pos);

            frames.Add(new FrameRec
            {
                frame = frameIdx,
                t = Time.time - startTime,
                pos = pos,
                vel = vel,
                pixel = px,
                eot = false,
            });
            frameIdx++;
        }

        // Project via the same pinhole (E, fx, fy, cx, cy) we export, so the
        // saved (u,v) is reproducible from (xyz, camera.json) by any consumer.
        // worldToCamera is OpenGL-style: camera looks down -Z in cam space; a
        // visible point has z_cam < 0.
        Vector2 ProjectToPixel(Vector3 world)
        {
            if (mainCam == null) return new Vector2(-1f, -1f);
            Vector4 ph = new Vector4(world.x, world.y, world.z, 1f);
            Vector4 pc = camSnap.worldToCamera * ph;
            if (pc.z >= 0f) return new Vector2(-1f, -1f);   // behind camera
            float u = camSnap.cx + camSnap.fx * (pc.x / -pc.z);
            float v = camSnap.cy - camSnap.fy * (pc.y / -pc.z);
            return new Vector2(u, v);
        }

        public EpisodeStats WriteEpisode(string outDir, int episodeId, string split, List<int> hitFrames)
        {
            var stats = new EpisodeStats
            {
                episodeId = episodeId,
                split = split,
                numStrokes = hitFrames.Count,
                numFrames = frames.Count,
                durationSec = frames.Count * Time.fixedDeltaTime,
                fixedTimestep = Time.fixedDeltaTime,
                hitFrames = new List<int>(hitFrames),
                bboxMin = new Vector3(Mathf.Infinity, Mathf.Infinity, Mathf.Infinity),
                bboxMax = new Vector3(Mathf.NegativeInfinity, Mathf.NegativeInfinity, Mathf.NegativeInfinity),
            };

            string dir = Path.Combine(outDir, split, $"ep_{episodeId:D06}");
            Directory.CreateDirectory(dir);

            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.AppendLine("frame,time,x,y,z,vx,vy,vz,u,v,eot");
            foreach (var f in frames)
            {
                float yo = f.pos.y - groundOffsetY;
                sb.AppendFormat(ci,
                    "{0},{1:F6},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F4},{9:F4},{10}\n",
                    f.frame, f.t, f.pos.x, yo, f.pos.z,
                    f.vel.x, f.vel.y, f.vel.z, f.pixel.x, f.pixel.y, f.eot ? 1 : 0);
                var posOff = new Vector3(f.pos.x, yo, f.pos.z);
                stats.bboxMin = Vector3.Min(stats.bboxMin, posOff);
                stats.bboxMax = Vector3.Max(stats.bboxMax, posOff);
                if (yo > stats.maxHeight) stats.maxHeight = yo;
            }
            File.WriteAllText(Path.Combine(dir, "frames.csv"), sb.ToString());
            WriteMetaJson(Path.Combine(dir, "meta.json"), stats);
            WriteCameraJson(Path.Combine(dir, "camera.json"));
            return stats;
        }

        void WriteCameraJson(string path)
        {
            var ci = CultureInfo.InvariantCulture;
            // Translate camera by -groundOffsetY in world Y so the saved camera
            // is consistent with the saved positions (which have the same offset
            // applied). Rotation/intrinsics are unchanged.
            var posOff = new Vector3(camSnap.position.x,
                                     camSnap.position.y - groundOffsetY,
                                     camSnap.position.z);
            var w2cOff = camSnap.worldToCamera * Matrix4x4.Translate(new Vector3(0f, groundOffsetY, 0f));
            var c2wOff = Matrix4x4.Translate(new Vector3(0f, -groundOffsetY, 0f)) * camSnap.cameraToWorld;

            var sb = new StringBuilder();
            sb.Append("{\n");
            sb.AppendFormat(ci, "  \"width\": {0},\n",  camSnap.width);
            sb.AppendFormat(ci, "  \"height\": {0},\n", camSnap.height);
            sb.AppendFormat(ci, "  \"fovYDeg\": {0:F6},\n", camSnap.fovYDeg);
            sb.AppendFormat(ci, "  \"fx\": {0:F6}, \"fy\": {1:F6}, \"cx\": {2:F6}, \"cy\": {3:F6},\n",
                camSnap.fx, camSnap.fy, camSnap.cx, camSnap.cy);
            sb.AppendFormat(ci, "  \"position\": [{0:F6}, {1:F6}, {2:F6}],\n",
                posOff.x, posOff.y, posOff.z);
            sb.AppendFormat(ci, "  \"rotationQuat_xyzw\": [{0:F6}, {1:F6}, {2:F6}, {3:F6}],\n",
                camSnap.rotation.x, camSnap.rotation.y, camSnap.rotation.z, camSnap.rotation.w);
            sb.AppendFormat(ci, "  \"groundOffsetY\": {0:F6},\n", groundOffsetY);
            sb.Append("  \"worldToCamera\": [\n");
            AppendMat4(sb, ci, w2cOff);
            sb.Append("  ],\n");
            sb.Append("  \"cameraToWorld\": [\n");
            AppendMat4(sb, ci, c2wOff);
            sb.Append("  ],\n");
            sb.Append("  \"convention\": \"unity_worldToCamera_opengl_view (cam looks down -Z, +Y up); positions and camera both translated by -groundOffsetY\"\n");
            sb.Append("}\n");
            File.WriteAllText(path, sb.ToString());
        }

        static void AppendMat4(StringBuilder sb, CultureInfo ci, Matrix4x4 m)
        {
            for (int r = 0; r < 4; r++)
            {
                sb.AppendFormat(ci, "    [{0:F6}, {1:F6}, {2:F6}, {3:F6}]{4}\n",
                    m[r, 0], m[r, 1], m[r, 2], m[r, 3], r < 3 ? "," : "");
            }
        }

        static void WriteMetaJson(string path, EpisodeStats s)
        {
            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.Append("{\n");
            sb.AppendFormat(ci, "  \"episodeId\": {0},\n", s.episodeId);
            sb.AppendFormat(ci, "  \"split\": \"{0}\",\n", s.split);
            sb.AppendFormat(ci, "  \"numStrokes\": {0},\n", s.numStrokes);
            sb.AppendFormat(ci, "  \"numFrames\": {0},\n", s.numFrames);
            sb.AppendFormat(ci, "  \"durationSec\": {0:F4},\n", s.durationSec);
            sb.AppendFormat(ci, "  \"fixedTimestep\": {0:F4},\n", s.fixedTimestep);
            sb.AppendFormat(ci, "  \"maxHeight\": {0:F4},\n", s.maxHeight);
            sb.AppendFormat(ci, "  \"bboxMin\": [{0:F4}, {1:F4}, {2:F4}],\n", s.bboxMin.x, s.bboxMin.y, s.bboxMin.z);
            sb.AppendFormat(ci, "  \"bboxMax\": [{0:F4}, {1:F4}, {2:F4}],\n", s.bboxMax.x, s.bboxMax.y, s.bboxMax.z);
            sb.Append("  \"source\": \"play_mode_game_bots\",\n");
            sb.Append("  \"hitFrames\": [");
            for (int i = 0; i < s.hitFrames.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(s.hitFrames[i]);
            }
            sb.Append("]\n}\n");
            File.WriteAllText(path, sb.ToString());
        }
    }

    public class EpisodeStats
    {
        public int episodeId;
        public string split;
        public int numStrokes;
        public int numFrames;
        public float durationSec;
        public float fixedTimestep;
        public float maxHeight;
        public Vector3 bboxMin;
        public Vector3 bboxMax;
        public List<int> hitFrames = new List<int>();
    }
}
