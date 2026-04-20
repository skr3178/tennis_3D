using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using TennisDataCollection.Runtime;

namespace TennisDataCollection.EditorTools
{
    // CLI entry: builds the impulse scene and enters Play Mode.
    //   Unity -projectPath <proj> -batchmode -nographics \
    //     -executeMethod TennisDataCollection.EditorTools.ImpulseDatasetRunner.Run \
    //     -episodes 10 -split test -outDir /abs/path/TennisDataset_impulse -seed 1337
    public static class ImpulseDatasetRunner
    {
        public static void Run()
        {
            int episodes = 10;
            string split = "test";
            int seed = 1337;
            string outDir = Path.Combine(Directory.GetCurrentDirectory(), "TennisDataset_impulse");

            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-episodes" && i + 1 < args.Length) int.TryParse(args[i + 1], out episodes);
                if (args[i] == "-split"    && i + 1 < args.Length) split = args[i + 1];
                if (args[i] == "-seed"     && i + 1 < args.Length) int.TryParse(args[i + 1], out seed);
                if (args[i] == "-outDir"   && i + 1 < args.Length) outDir = args[i + 1];
            }
            Debug.Log($"[ImpulseDatasetRunner] episodes={episodes} split={split} seed={seed} outDir={outDir}");

            ImpulseSceneBuilder.Build();
            var scene = EditorSceneManager.OpenScene(ImpulseSceneBuilder.OutputScene, OpenSceneMode.Single);

            var runnerGo = GameObject.Find("ImpulseRunner");
            if (runnerGo == null)
            {
                Debug.LogError("[ImpulseDatasetRunner] ImpulseRunner not found");
                EditorApplication.Exit(2);
                return;
            }
            var runner = runnerGo.GetComponent<ImpulseEpisodeRunner>();
            runner.episodesToRun = episodes;
            runner.split = split;
            runner.outDir = outDir;
            runner.seed = seed;

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            EditorApplication.EnterPlaymode();
        }
    }
}
