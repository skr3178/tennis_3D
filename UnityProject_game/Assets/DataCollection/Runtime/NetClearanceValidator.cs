using UnityEngine;

namespace TennisDataCollection.Runtime
{
    [RequireComponent(typeof(Collider))]
    public class NetClearanceValidator : MonoBehaviour
    {
        public GameEpisodeRunner runner;

        void OnCollisionEnter(Collision collision)
        {
            if (runner == null) return;
            if (!collision.collider.CompareTag("Ball")) return;
            runner.InvalidateCurrentEpisode("ball_hit_net");
        }

        void OnTriggerEnter(Collider other)
        {
            if (runner == null) return;
            if (!other.CompareTag("Ball")) return;
            runner.InvalidateCurrentEpisode("ball_hit_net");
        }
    }
}
