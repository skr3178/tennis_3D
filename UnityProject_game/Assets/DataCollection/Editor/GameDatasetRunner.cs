using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using TennisDataCollection.Runtime;

namespace TennisDataCollection.EditorTools
{
    // CLI entry point. Builds the Collection scene, wires runner parameters from
    // command-line args, then enters Play Mode. The runtime EpisodeRunner calls
    // EditorApplication.Exit when all episodes are saved.
    //
    // Usage:
    //   Unity -projectPath <proj> -batchmode -nographics \
    //         -executeMethod TennisDataCollection.EditorTools.GameDatasetRunner.Run \
    //         -episodes 10 -split test -outDir /abs/path/TennisDataset_game -seed 1337
    //
    // NOTE: do NOT pass -quit; the process must stay alive for Play Mode.
    public static class GameDatasetRunner
    {
        public static void Run()
        {
            int episodes = 10;
            string split = "test";
            int seed = 1337;
            int episodeIdOffset = 0;
            string outDir = Path.Combine(Directory.GetCurrentDirectory(), "TennisDataset_game");

            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-episodes" && i + 1 < args.Length) int.TryParse(args[i + 1], out episodes);
                if (args[i] == "-split"    && i + 1 < args.Length) split = args[i + 1];
                if (args[i] == "-seed"     && i + 1 < args.Length) int.TryParse(args[i + 1], out seed);
                if (args[i] == "-outDir"   && i + 1 < args.Length) outDir = args[i + 1];
                if (args[i] == "-episodeIdOffset" && i + 1 < args.Length) int.TryParse(args[i + 1], out episodeIdOffset);
            }
            Debug.Log($"[GameDatasetRunner] episodes={episodes} split={split} seed={seed} episodeIdOffset={episodeIdOffset} outDir={outDir}");

            DisableStaleSceneBackups();
            GameSceneBuilder.Build();  // always rebuild to pick up any script changes
            var scene = EditorSceneManager.OpenScene(GameSceneBuilder.OutputScene, OpenSceneMode.Single);

            var runnerGo = GameObject.Find("EpisodeRunner");
            if (runnerGo == null)
            {
                Debug.LogError("[GameDatasetRunner] EpisodeRunner not found in scene");
                EditorApplication.Exit(2);
                return;
            }
            var runner = runnerGo.GetComponent<GameEpisodeRunner>();
            runner.episodesToRun = episodes;
            runner.split = split;
            runner.outDir = outDir;
            runner.seed = seed;
            runner.episodeIdOffset = episodeIdOffset;

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);

            EditorApplication.EnterPlaymode();
        }

        static void DisableStaleSceneBackups()
        {
            string backupDir = Path.Combine(Directory.GetCurrentDirectory(), "Temp", "__Backupscenes");
            if (!Directory.Exists(backupDir)) return;

            string disabledDir = backupDir + "_disabled";
            int suffix = 0;
            while (Directory.Exists(disabledDir))
            {
                suffix++;
                disabledDir = backupDir + "_disabled_" + suffix.ToString("D2");
            }

            Directory.Move(backupDir, disabledDir);
            Debug.Log($"[GameDatasetRunner] sidelined stale scene backups -> {disabledDir}");
        }
    }
}
