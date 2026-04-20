using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace TennisDataCollection.Runtime
{
    // Auto-attaches a TrajectoryRecorder to the Ball when Play Mode starts in any
    // scene, then flushes a CSV when Play Mode stops (or the app quits). Drop-in:
    // no scene edits required.
    public static class AutoPlayRecorderBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Init()
        {
            if (Application.isBatchMode) return;
            if (GameObject.Find("EpisodeRunner") != null) return;

            var ball = GameObject.FindWithTag("Ball");
            if (ball == null) ball = GameObject.Find("ball");
            if (ball == null)
            {
                Debug.LogWarning("[AutoPlayRecorder] no Ball-tagged GameObject found in scene");
                return;
            }

            var recorder = ball.GetComponent<TrajectoryRecorder>();
            if (recorder == null) recorder = ball.AddComponent<TrajectoryRecorder>();
            if (Camera.main != null) recorder.SetCamera(Camera.main);
            // Tutorial scene's court surface sits at world y ≈ 1.295.
            // Subtract so saved CSV uses paper convention (court at y=0).
            recorder.SetGroundOffsetY(1.295f);
            recorder.BeginRecording();

            var hookGo = new GameObject("__AutoPlayRecorderHook");
            Object.DontDestroyOnLoad(hookGo);
            var hook = hookGo.AddComponent<AutoPlayRecorderHook>();
            hook.recorder = recorder;

            Debug.Log($"[AutoPlayRecorder] recording started on '{ball.name}'");
        }
    }

    public class AutoPlayRecorderHook : MonoBehaviour
    {
        public TrajectoryRecorder recorder;
        bool saved;

        void OnApplicationQuit() => Save();
        void OnDisable() => Save();

        void Save()
        {
            if (saved || recorder == null) return;
            saved = true;
            recorder.EndRecording(markLastAsEoT: true);

            string sessionTag = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string outRoot = Path.Combine(Directory.GetCurrentDirectory(), "GuiPlaySession");
            var stats = recorder.WriteEpisode(outRoot, 0, sessionTag, new List<int>());

            Debug.Log($"[AutoPlayRecorder] saved {stats.numFrames} frames " +
                      $"bbox=({stats.bboxMin.x:F2},{stats.bboxMin.y:F2},{stats.bboxMin.z:F2}) -> " +
                      $"({stats.bboxMax.x:F2},{stats.bboxMax.y:F2},{stats.bboxMax.z:F2}) " +
                      $"maxH={stats.maxHeight:F2}m  -> {outRoot}/{sessionTag}/ep_000000/");
        }
    }
}
