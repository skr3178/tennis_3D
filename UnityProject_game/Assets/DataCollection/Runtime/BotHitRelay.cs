using UnityEngine;

namespace TennisDataCollection.Runtime
{
    // Attached alongside the existing Bot.cs on each bot GameObject.
    // Bot.cs already has its own OnTriggerEnter; Unity delivers the message to
    // every MonoBehaviour on the GameObject, so this runs in parallel — no
    // modification of Bot.cs required.
    public class BotHitRelay : MonoBehaviour
    {
        public GameEpisodeRunner runner;

        void OnTriggerEnter(Collider other)
        {
            if (runner == null) return;
            if (!other.CompareTag("Ball")) return;
            var paperBot = GetComponent<PaperTennisBot>();
            if (paperBot == null) return;
            runner.OnBotHit(paperBot, paperBot.transform.position);
        }
    }
}
