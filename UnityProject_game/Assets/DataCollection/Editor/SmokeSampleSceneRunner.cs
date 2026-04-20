using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using TennisDataCollection.Runtime;

namespace TennisDataCollection.EditorTools
{
    // Opens the PRISTINE tutorial SampleScene, attaches TrajectoryRecorder to the
    // Ball, runs Play Mode for -seconds wall-time, writes one CSV of the whole
    // continuous session into -outDir (episode id 0). No scene modifications are
    // saved; this is a read-only smoke probe of the tutorial's physics/gameplay.
    //
    // Usage:
    //   Unity -projectPath <proj> -batchmode -nographics \
    //         -executeMethod TennisDataCollection.EditorTools.SmokeSampleSceneRunner.Run \
    //         -seconds 60 -outDir /abs/path/SmokeSampleScene
    //
    // Do NOT pass -quit.
    public static class SmokeSampleSceneRunner
    {
        const string ScenePath = "Assets/Scenes/SampleScene.unity";

        static TrajectoryRecorder recorder;
        static float playStartRealtime;
        static float runSeconds = 60f;
        static string outDir;

        public static void Run()
        {
            outDir = Path.Combine(Directory.GetCurrentDirectory(), "SmokeSampleScene");
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-seconds" && i + 1 < args.Length) float.TryParse(args[i + 1], out runSeconds);
                if (args[i] == "-outDir"  && i + 1 < args.Length) outDir = args[i + 1];
            }
            Debug.Log($"[Smoke] scene={ScenePath} seconds={runSeconds} outDir={outDir}");

            // Survive Unity's domain reload on play-mode entry: skip the reload so
            // our static state (registered callbacks) stays alive across the boundary.
            EditorSettings.enterPlayModeOptionsEnabled = true;
            EditorSettings.enterPlayModeOptions = EnterPlayModeOptions.DisableDomainReload;

            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
            EditorApplication.EnterPlaymode();
        }

        static void OnPlayModeChanged(PlayModeStateChange s)
        {
            if (s == PlayModeStateChange.EnteredPlayMode)
            {
                var ball = GameObject.FindWithTag("Ball");
                if (ball == null) ball = GameObject.Find("ball");
                if (ball == null)
                {
                    Debug.LogError("[Smoke] no Ball-tagged object found");
                    EditorApplication.ExitPlaymode();
                    return;
                }
                Debug.Log($"[Smoke] found ball: {ball.name} at {ball.transform.position}");
                recorder = ball.GetComponent<TrajectoryRecorder>();
                if (recorder == null) recorder = ball.AddComponent<TrajectoryRecorder>();
                recorder.BeginRecording();
                playStartRealtime = Time.realtimeSinceStartup;
                EditorApplication.update += Tick;
            }
            else if (s == PlayModeStateChange.ExitingPlayMode)
            {
                EditorApplication.update -= Tick;
                if (recorder != null)
                {
                    recorder.EndRecording(markLastAsEoT: true);
                    var stats = recorder.WriteEpisode(outDir, 0, "test", new List<int>());
                    Debug.Log($"[Smoke] wrote ep_000000 frames={stats.numFrames} " +
                              $"bboxMin=({stats.bboxMin.x:F2},{stats.bboxMin.y:F2},{stats.bboxMin.z:F2}) " +
                              $"bboxMax=({stats.bboxMax.x:F2},{stats.bboxMax.y:F2},{stats.bboxMax.z:F2}) " +
                              $"maxH={stats.maxHeight:F2} -> {outDir}");
                }
                EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            }
            else if (s == PlayModeStateChange.EnteredEditMode)
            {
                EditorApplication.Exit(0);
            }
        }

        static void Tick()
        {
            if (!EditorApplication.isPlaying) return;
            if (Time.realtimeSinceStartup - playStartRealtime > runSeconds)
            {
                EditorApplication.ExitPlaymode();
            }
        }
    }
}
