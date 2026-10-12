package com.axiomcloud.repository;

import com.axiomcloud.model.TrainedModelEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TrainedModelRepository extends JpaRepository<TrainedModelEntity, Integer> {

    List<TrainedModelEntity> findByJobIdOrderByCreatedAtDesc(String jobId);

    List<TrainedModelEntity> findAllByOrderByCreatedAtDesc();

    List<TrainedModelEntity> findByIsDeployedTrue();

    long countByIsDeployedTrue();

    void deleteByJobId(String jobId);

    @Modifying
    @Query("UPDATE TrainedModelEntity m SET m.isDeployed = false WHERE m.jobId = :jobId AND m.id != :modelId")
    void undeployOtherModelsInJob(String jobId, Integer modelId);
}
